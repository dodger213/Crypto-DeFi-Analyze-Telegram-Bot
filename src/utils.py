import random
import logging
from datetime import datetime, timedelta

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from data.models import User
from config import SUBSCRIPTION_WALLET_ADDRESS

from services.blockchain import * 
from services.analytics import *
from services.notification import *
from services.user_management import *

async def check_callback_user(update: Update) -> User:
    """Check if user exists in database, create if not, and update activity"""
    return await get_or_create_user(
        user_id=update.callback_query.from_user.id,
        username=update.callback_query.from_user.username,
        first_name=update.callback_query.from_user.first_name,
        last_name=update.callback_query.from_user.last_name
    )

async def check_premium_required(update: Update, context: ContextTypes.DEFAULT_TYPE, feature_name: str) -> bool:
    """Check if a premium feature is being accessed by a non-premium user"""
    user = await check_callback_user(update)
    
    if not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="back")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"⭐ <b>Premium Feature</b>\n\n"
            f"The {feature_name} feature is only available to premium users.\n\n"
            f"💎 Upgrade to premium to unlock all features!",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return True
    
    return False

async def send_premium_welcome_message(update: Update, context: ContextTypes.DEFAULT_TYPE, user: User, plan: str, premium_until: datetime) -> None:
    """Send a welcome message with premium tips to new premium users"""
    welcome_message = (
        f"🎉 <b>Welcome to DeFi-Scope Premium!</b>\n\n"
        f"Hi {user.first_name}, thank you for upgrading to our premium service.\n\n"
        f"<b>Here are some premium features you can now access:</b>\n\n"
        f"• <b>Unlimited Token & Wallet Scans</b>\n"
        f"  Use /scan_token and /scan_wallet as much as you need\n\n"
        f"• <b>Deployer Wallet Analysis</b>\n"
        f"  Use /dw [contract] to analyze token deployers\n\n"
        f"• <b>Top Holders & Whale Tracking</b>\n"
        f"  Use /th [contract] to see top token holders\n\n"
        f"• <b>Wallet & Token Tracking</b>\n"
        f"  Use /track commands to monitor wallets and tokens\n\n"
        f"Need help with premium features? Type /premium_help anytime!\n\n"
        f"Your {plan} subscription is active until: <b>{premium_until.strftime('%d %B %Y')}</b>"
    )
    
    # Send as a new message to avoid replacing the payment confirmation
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=welcome_message,
        parse_mode=ParseMode.HTML
    )
    """Handle payment retry callback"""
    query = update.callback_query
    
    # Clear the stored transaction ID
    if "transaction_id" in context.user_data:
        del context.user_data["transaction_id"]
    
    # Set up to collect a new transaction ID
    context.user_data["awaiting_transaction_id"] = True
    context.user_data["premium_plan"] = plan
    
    keyboard = [[InlineKeyboardButton("🔙 Cancel", callback_data="back")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "📝 <b>New Transaction ID Required</b>\n\n"
        "Please send the new transaction hash/ID of your payment.\n\n"
        "You can find this in your wallet's transaction history after sending the payment.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )

async def handle_token_analysis_input(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    analysis_type: str,
    get_data_func,
    format_response_func,
    scan_count_type: str,
    processing_message_text: str,
    error_message_text: str,
    no_data_message_text: str
) -> None:
    """
    Generic handler for token analysis inputs
    
    Args:
        update: The update object
        context: The context object
        analysis_type: Type of analysis being performed (for logging)
        get_data_func: Function to get the data (takes token_address)
        format_response_func: Function to format the response (takes data and token_data)
        scan_count_type: Type of scan to increment count for
        processing_message_text: Text to show while processing
        error_message_text: Text to show on error
        no_data_message_text: Text to show when no data is found
    """
    token_address = update.message.text.strip()
    
    # Validate address
    if not await is_valid_address(token_address):
        # Add back button to validation error message
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "⚠️ Please provide a valid Ethereum address or token contract address.",
            reply_markup=reply_markup
        )
        return
    
    # Send processing message
    processing_message = await update.message.reply_text(processing_message_text)
    
    try:
        # Get data
        data = await get_data_func(token_address)
        token_data = await get_token_data(token_address)
        
        if not data or not token_data:
            # Add back button when no data is found
            keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await processing_message.edit_text(
                no_data_message_text,
                reply_markup=reply_markup
            )
            return
        
        # Format the response
        response, keyboard = format_response_func(data, token_data, token_address)
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        success = False
        try:
            # Try to edit the current message
            await processing_message.edit_text(
                response,
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
            success = True
        except Exception as e:
            logging.error(f"Error in handle_{analysis_type}: {e}")
            # If editing fails, send a new message
            await update.message.reply_text(
                response,
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
            success = True
            # Delete the original message if possible
            try:
                await processing_message.delete()
            except:
                pass
        
        # Only increment scan count if we successfully displayed data
        if success:
            # Get the user directly from the message update
            user_id = update.effective_user.id
            user = get_user(user_id)
            if not user:
                # Create user if not exists
                user = User(user_id=user_id, username=update.effective_user.username)
                # Save user to database if needed
            
            await increment_scan_count(user_id, scan_count_type)
    
    except Exception as e:
        logging.error(f"Error in handle_expected_input ({analysis_type}): {e}")
        
        # Add back button to exception error message
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await processing_message.edit_text(
            error_message_text,
            reply_markup=reply_markup
        )

async def get_token_data(token_address: str) -> dict:
    """
    Placeholder function for getting token data
    
    Args:
        token_address: The token contract address
    
    Returns:
        Dictionary containing token data
    """

    logging.info(f"Placeholder: get_token_data called for {token_address}")
    
    # Generate random token data for demonstration purposes
    token_symbols = ["USDT", "WETH", "PEPE", "SHIB", "DOGE", "LINK", "UNI", "AAVE", "COMP", "SNX"]
    token_names = ["Tether", "Wrapped Ethereum", "Pepe", "Shiba Inu", "Dogecoin", "Chainlink", "Uniswap", "Aave", "Compound", "Synthetix"]
    
    # Pick a random name and symbol
    index = random.randint(0, len(token_symbols) - 1)
    symbol = token_symbols[index]
    name = token_names[index]
    
    # Generate random price and market cap
    current_price = round(random.uniform(0.00001, 100), 6)
    market_cap = round(current_price * random.uniform(1000000, 10000000000), 2)
    
    # Generate random ATH data
    ath_multiplier = random.uniform(1.5, 10)
    ath_price = round(current_price * ath_multiplier, 6)
    ath_market_cap = round(market_cap * ath_multiplier, 2)
    
    # Generate random dates
    now = datetime.now()
    launch_date = (now - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d")
    ath_date = (now - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
    
    # Create token data dictionary
    token_data = {
        "address": token_address,
        "name": name,
        "symbol": symbol,
        "current_price": current_price,
        "current_market_cap": market_cap,
        "holders_count": random.randint(100, 10000),
        "liquidity": round(random.uniform(10000, 1000000), 2),
        "launch_date": launch_date,
        "ath_price": ath_price,
        "ath_date": ath_date,
        "ath_market_cap": ath_market_cap,
        "deployer_wallet": {
            "address": "0x" + ''.join(random.choices('0123456789abcdef', k=40)),
            "tokens_deployed": random.randint(1, 20),
            "success_rate": round(random.uniform(10, 100), 2),
            "avg_roi": round(random.uniform(-50, 500), 2),
            "rugpull_count": random.randint(0, 5),
            "risk_level": random.choice(["Low", "Medium", "High", "Very High"])
        }
    }
    
    return token_data

def format_first_buyers_response(first_buyers: List[Dict[str, Any]], 
                                token_data: Dict[str, Any], 
                                token_address: str) -> Tuple[str, List[List[InlineKeyboardButton]]]:
    """
    Format the response for first buyers analysis
    
    Args:
        first_buyers: List of first buyer data
        token_data: Token information
        token_address: The token address
        
    Returns:
        Tuple of (formatted response text, keyboard buttons)
    """
    response = (
        f"🛒 <b>First Buyers Analysis for {token_data.get('name', 'Unknown Token')} ({token_data.get('symbol', 'N/A')})</b>\n\n"
        f"Contract: `{token_address}`\n\n"
    )
    
    for i, buyer in enumerate(first_buyers[:10], 1):
        response += (
            f"{i}. `{buyer['address'][:6]}...{buyer['address'][-4:]}`\n"
            f"   Buy Amount: {buyer.get('buy_amount', 'N/A')} tokens\n"
            f"   Buy Value: ${buyer.get('buy_value', 'N/A')}\n"
            f"   Current PNL: {buyer.get('pnl', 'N/A')}%\n\n"
        )
    
    keyboard = [
        [
            InlineKeyboardButton("Export Data", callback_data=f"export_fb_{token_address}"),
            InlineKeyboardButton("Track Token", callback_data=f"track_token_{token_address}")
        ],
        [InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]
    ]
    
    return response, keyboard

def format_profitable_wallets_response(profitable_wallets: List[Dict[str, Any]], 
                                      token_data: Dict[str, Any], 
                                      token_address: str) -> Tuple[str, List[List[InlineKeyboardButton]]]:
    """
    Format the response for most profitable wallets analysis
    
    Args:
        profitable_wallets: List of profitable wallet data
        token_data: Token information
        token_address: The token address
        
    Returns:
        Tuple of (formatted response text, keyboard buttons)
    """
    response = (
        f"💰 <b>Most Profitable Wallets for {token_data.get('name', 'Unknown Token')} ({token_data.get('symbol', 'N/A')})</b>\n\n"
        f"Contract: `{token_address}`\n\n"
    )
    
    for i, wallet in enumerate(profitable_wallets[:10], 1):
        response += (
            f"{i}. `{wallet['address'][:6]}...{wallet['address'][-4:]}`\n"
            f"   Buy Amount: {wallet.get('buy_amount', 'N/A')} tokens\n"
            f"   Sell Amount: {wallet.get('sell_amount', 'N/A')} tokens\n"
            f"   Profit: ${wallet.get('profit', 'N/A')}\n"
            f"   ROI: {wallet.get('roi', 'N/A')}%\n\n"
        )
    
    keyboard = [
        [
            InlineKeyboardButton("Export Data", callback_data=f"export_mpw_{token_address}"),
            InlineKeyboardButton("Track Token", callback_data=f"track_token_{token_address}")
        ],
        [InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]
    ]
    
    return response, keyboard