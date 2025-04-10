import logging
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from config import FREE_TOKEN_SCANS_DAILY, FREE_RESPONSE_DAILY, FREE_WALLET_SCANS_DAILY, PREMIUM_RESPONSE_DAILY
from data.database import ( get_user_tracking_subscriptions, get_user)
from data.models import User, TrackingSubscription
from data.database import *

from services.blockchain import *
from services.notification import *
from services.user_management import *
from services.payment import *

from utils import *

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all callback queries from inline keyboards"""
    query = update.callback_query
    await query.answer()  
    
    callback_data = query.data
    
    logging.info(f"Callback query received: {callback_data}")
    
    if callback_data == "start_menu" or callback_data == "main_menu":
        await handle_start_menu(update, context)
    elif callback_data == "back":
        current_text = query.message.text or query.message.caption or ""
        if "Welcome to Crypto DeFi Analyze Bot" in current_text and "Your Ultimate DeFi Intelligence Bot" in current_text:
            await query.answer("You're already at the main menu")
        else:
            await handle_start_menu(update, context)
    elif callback_data == "select_network":
        await handle_select_network(update, context)
    elif callback_data.startswith("set_default_network_"):
        await handle_set_default_network(update, context)
    elif callback_data == "token_analysis":
        await handle_token_analysis(update, context)
    elif callback_data == "wallet_analysis":
        await handle_wallet_analysis(update, context)
    elif callback_data == "tracking_and_monitoring":
        await handle_tracking_and_monitoring(update, context)
    elif callback_data == "kol_wallets":
        await handle_kol_wallets(update, context)
    elif callback_data == "token_first_buyers":
        await handle_first_buyers(update, context)
    elif callback_data == "token_most_profitable_wallets":
        await handle_token_most_profitable_wallets(update, context)
    elif callback_data == "token_ath":
        await handle_ath(update, context)
    elif callback_data == "token_deployer_wallet_scan":
        await handle_deployer_wallet_scan(update, context)
    elif callback_data == "token_top_holders":
        await handle_top_holders(update, context)
    elif callback_data.startswith("setup_whale_tracking_"):
        await handle_setup_whale_tracking(update, context)
    elif callback_data == "token_high_net_worth_holders":
        await handle_high_net_worth_holders(update, context)
    elif callback_data == "wallet_most_profitable_in_period":
        await handle_wallet_most_profitable_in_period(update, context)
    elif callback_data == "wallet_holding_duration":
        await handle_wallet_holding_duration(update, context)
    elif callback_data == "most_profitable_token_deployer_wallet":
        await handle_most_profitable_token_deployer_wallet(update, context)
    elif callback_data == "tokens_deployed_by_wallet":
        await handle_tokens_deployed_by_wallet(update, context)
    elif callback_data == "track_wallet_buy_sell":
        await handle_track_wallet_buy_sell(update, context)
    elif callback_data == "track_new_token_deploy":
        await handle_track_new_token_deploy(update, context)
    elif callback_data == "track_profitable_wallets":
        await handle_track_profitable_wallets(update, context)
    elif callback_data == "kol_wallet_profitability":
        await handle_kol_wallet_profitability(update, context)
    elif callback_data == "track_whale_wallets":
        await handle_track_whale_wallets(update, context)
    elif "_chain_" in callback_data:
        await handle_chain_selection_callback(update, context)
    elif callback_data.startswith("kol_period_"):
        days = int(callback_data.replace("kol_period_", ""))
        context.user_data["selected_period"] = days
        await handle_kol_period_selection(update, context)
    elif callback_data == "view_tracking_subscriptions":
        await handle_view_tracking_subscriptions(update, context)
    elif callback_data == "manage_wallet_tracking":
        await handle_manage_wallet_tracking(update, context)
    elif callback_data == "manage_deployment_tracking":
        await handle_manage_deployment_tracking(update, context)
    elif callback_data == "manage_token_tracking":
        await handle_manage_token_tracking(update, context)
    elif callback_data.startswith("remove_tracking_"):
        target_address = callback_data.replace("remove_tracking_", "")
        await handle_remove_tracking(update, context, target_address)
    elif callback_data == "general_help":
        await handle_general_help(update, context)
    elif callback_data == "token_analysis_help":
        await handle_token_analysis_help(update, context)
    elif callback_data == "wallet_analysis_help":
        await handle_wallet_analysis_help(update, context)    
    elif callback_data == "tracking_and_monitoring_help":
        await handle_tracking_and_monitoring_help(update, context)
    elif callback_data == "kol_wallets_help":
        await handle_kol_wallets_help(update, context)
    elif callback_data == "premium_info":
        await handle_premium_info(update, context)
    elif callback_data.startswith("premium_plan_"):
        parts = callback_data.replace("premium_plan_", "").split("_")
        if len(parts) == 2:
            plan, currency = parts
            await handle_premium_purchase(update, context, plan, currency)
        else:
            await query.answer("Invalid plan selection", show_alert=True)
    elif callback_data.startswith("payment_made_"):
        parts = callback_data.replace("payment_made_", "").split("_")
        if len(parts) == 2:
            plan, currency = parts
            await handle_payment_made(update, context, plan, currency)
        else:
            await query.answer("Invalid payment confirmation", show_alert=True)
        
    else:
        await query.answer(
            "Sorry, I couldn't process that request. Please try again.", show_alert=True
        )

async def handle_expected_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle expected inputs from conversation states"""
    expecting = context.user_data.get("expecting")
 
    if not expecting:
        return
   
    del context.user_data["expecting"]

    if expecting == "first_buyers_token_address":
        await handle_token_analysis_input(
            update=update,
            context=context,
            analysis_type="first_buyers",
            get_data_func=get_token_first_buyers,
            format_response_func=format_first_buyers_response,
            scan_count_type="first_buy_wallet_scan",
            processing_message_text="🔍 Analyzing token's first buyers... This may take a moment.",
            error_message_text="❌ An error occurred while analyzing the token's first buyers. Please try again later.",
            no_data_message_text="❌ Could not find first buyers data for this token."
        )
        
    elif expecting == "token_most_profitable_wallets_token_address":
        await handle_token_analysis_input(
            update=update,
            context=context,
            analysis_type="most_profitable_wallets",
            get_data_func=get_token_profitable_wallets,
            format_response_func=format_profitable_wallets_response,
            scan_count_type="token_most_profitable_wallet_scan",
            processing_message_text="🔍 Analyzing most profitable wallets for this token... This may take a moment.",
            error_message_text="❌ An error occurred while analyzing the token's most profitable wallets. Please try again later.",
            no_data_message_text="❌ Could not find profitable wallets data for this token."
        )
    
    elif expecting == "ath_token_address":
        await handle_token_analysis_input(
            update=update,
            context=context,
            analysis_type="ath",
            get_data_func=get_ath_data,
            format_response_func=format_ath_response,
            scan_count_type="ath_scan",
            processing_message_text="🔍 Analyzing token's ATH data... This may take a moment.",
            error_message_text="❌ An error occurred while analyzing the token's ATH data. Please try again later.",
            no_data_message_text="❌ Could not find ATH data for this token."
        )

    elif expecting == "deployer_wallet_scan_token":
        await handle_token_analysis_input(
            update=update,
            context=context,
            analysis_type="deployer_wallet_scan",
            get_data_func=get_deployer_wallet_scan_data,
            format_response_func=format_deployer_wallet_scan_response,
            scan_count_type="deployer_wallet_scan",
            processing_message_text="🔍 Analyzing token deployer wallet... This may take a moment.",
            error_message_text="❌ An error occurred while analyzing the deployer wallet. Please try again later.",
            no_data_message_text="❌ Could not find deployer wallet data for this token."
        )

    elif expecting == "top_holders_token_address":
        await handle_token_analysis_input(
            update=update,
            context=context,
            analysis_type="top_holders",
            get_data_func=get_token_top_holders,
            format_response_func=format_top_holders_response,
            scan_count_type="top_holders_scan",
            processing_message_text="🔍 Analyzing token's top holders... This may take a moment.",
            error_message_text="❌ An error occurred while analyzing the token's top holders. Please try again later.",
            no_data_message_text="❌ Could not find top holders data for this token."
        )

    elif expecting == "track_whale_wallets_token":
        token_address = update.message.text.strip()
        
        # Validate address format
        if not await is_valid_address(token_address):
            await update.message.reply_text(
                f"❌ <b>Invalid Token Address</b>\n\n"
                f"Please provide a valid Ethereum token address starting with 0x and containing 42 characters.",
                parse_mode=ParseMode.HTML
            )
            return
        
        # Send processing message
        processing_message = await update.message.reply_text(
            "🔍 Analyzing token's top holders and whale wallets... This may take a moment."
        )
        
        try:
            chain=context.user_data.get("default_network", "eth")
            token_info = await get_token_info(token_address, chain)
            
            if not token_info:
                await processing_message.edit_text(
                    "❌ Could not find information for this token. Please check the address and try again."
                )
                return
            
            # Get top holders for this token
            top_holders = await get_token_top_holders(token_address, limit=10)
            
            if not top_holders:
                await processing_message.edit_text(
                    "❌ Could not find top holders for this token at this time."
                )
                return
            
            # Get deployer wallet
            deployer_info = await get_deployer_wallet_scan_data(token_address, chain)
            deployer_wallet = deployer_info.get("deployer_address") if deployer_info else None
            
            # Create token tracking subscription with metadata
            token_subscription = TrackingSubscription(
                user_id=update.effective_user.id,
                tracking_type="token_whale_tracking",
                target_address=token_address,
                is_active=True,
                created_at=datetime.now(),
                metadata={
                    "token_name": token_info.get("name", "Unknown"),
                    "token_symbol": token_info.get("symbol", "Unknown"),
                    "deployer": deployer_wallet
                }
            )
            save_tracking_subscription(token_subscription)
            
            # Track deployer wallet if available
            if deployer_wallet:
                deployer_subscription = TrackingSubscription(
                    user_id=update.effective_user.id,
                    tracking_type="wallet_trades",
                    target_address=deployer_wallet,
                    is_active=True,
                    created_at=datetime.now(),
                    metadata={
                        "role": "deployer",
                        "token": token_address,
                        "token_name": token_info.get("name", "Unknown"),
                        "token_symbol": token_info.get("symbol", "Unknown")
                    }
                )
                save_tracking_subscription(deployer_subscription)
            
            # Track top holders
            for holder in top_holders:
                holder_subscription = TrackingSubscription(
                    user_id=update.effective_user.id,
                    tracking_type="wallet_trades",
                    target_address=holder["address"],
                    is_active=True,
                    created_at=datetime.now(),
                    metadata={
                        "role": "top_holder",
                        "token": token_address,
                        "token_name": token_info.get("name", "Unknown"),
                        "token_symbol": token_info.get("symbol", "Unknown"),
                        "percentage": holder.get("percentage", 0)
                    }
                )
                save_tracking_subscription(holder_subscription)
            
            # Format the response
            response = (
                f"✅ <b>Whale Tracking Set Up Successfully!</b>\n\n"
                f"You are now actively monitoring the behavior of whales and top holders for the token:\n"
                f"<b>{token_info.get('name', 'Unknown')}</b> ({token_info.get('symbol', 'N/A')})\n"
                f"`{token_address[:6]}...{token_address[-4:]}`\n\n"
                f"🔔 Stay ahead of sudden market shifts, early exits, and accumulation patterns to make smarter decisions based on real-time wallet activity!\n\n"

            )

            response = (
                f"✅ <b>Whale Tracking Has Been Successfully Set Up!</b>\n\n"
                f"You are now actively monitoring the behavior of whales and top holders for the token:\n"
                f"<b>{token_info.get('name', 'Unknown')}</b> (<code>{token_info.get('symbol', 'N/A')}</code>)\n"
                f"<code>{token_address[:6]}...{token_address[-4:]}</code>\n\n"
                f"This means you'll receive timely notifications whenever the developer, any of the top 10 holders, or major whale wallets make a significant move—"
                f"whether it's buying, selling, or offloading tokens.\n\n"
                f"🔔 Stay ahead of sudden market shifts, early exits, and accumulation patterns to make smarter decisions based on real-time wallet activity!"
            )
            
            if deployer_wallet:
                response += (
                    f"<b>🧑‍💻 Deployer Wallet:</b>\n"
                    f"`{deployer_wallet[:6]}...{deployer_wallet[-4:]}`\n\n"
                )
            
            response += "<b>🐳 Top Holders Being Tracked:</b>\n"
            
            for i, holder in enumerate(top_holders[:5], 1):  # Show first 5 for brevity
                percentage = holder.get('percentage', 0)
                response += (
                    f"{i}. `{holder['address'][:6]}...{holder['address'][-4:]}`\n"
                    f"   Holdings: {percentage:.2f}% of supply\n"
                )
            
            if len(top_holders) > 5:
                response += f"\n...and {len(top_holders) - 5} more holders\n"
            
            response += (
                f"\n<b>🚨 You will receive alerts when:</b>\n"
                f"• The developer sells tokens\n"
                f"• Any of the top 10 holders sell\n"
                f"• Any whale wallet dumps the token"
            )
            
            # Add button to view all tracking subscriptions
            keyboard = [
                [InlineKeyboardButton("👁️ View All Tracking", callback_data="view_tracking_subscriptions")],
                [InlineKeyboardButton("🔙 Back", callback_data="kol_wallets")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await processing_message.edit_text(
                response,
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
            
        except Exception as e:
            logging.error(f"Error in handle_whale_tracking_input: {e}")
            await processing_message.edit_text(
                "❌ An error occurred while setting up whale tracking. Please try again later."
            )

    
    elif expecting == "high_net_worth_holders_token_address":
        await handle_token_analysis_input(
            update=update,
            context=context,
            analysis_type="high_net_worth_holders",
            get_data_func=get_high_net_worth_holders,
            format_response_func=format_high_net_worth_holders_response,
            scan_count_type="high_net_worth_holders_scan",
            processing_message_text="🔍 Analyzing token's high net worth holders... This may take a moment.",
            error_message_text="❌ An error occurred while analyzing the token's high net worth holders. Please try again later.",
            no_data_message_text="❌ Could not find high net worth holders data for this token."
        ) 

    elif expecting == "wallet_holding_duration_address":
        await handle_wallet_analysis_input(
            update=update,
            context=context,
            analysis_type="wallet_holding_duration",
            get_data_func=get_wallet_holding_duration,
            format_response_func=format_wallet_holding_duration_response,
            scan_count_type="wallet_holding_duration_scan",
            processing_message_text="🔍 Analyzing wallet's token holding duration... This may take a moment.",
            error_message_text="❌ An error occurred while analyzing the wallet's holding duration. Please try again later.",
            no_data_message_text="❌ Could not find holding duration data for this wallet."
        )
    
    elif expecting == "tokens_deployed_wallet_address":
        await handle_wallet_analysis_input(
            update=update,
            context=context,
            analysis_type="tokens_deployed_by_wallet",
            get_data_func=get_tokens_deployed_by_wallet,
            format_response_func=format_tokens_deployed_response,
            scan_count_type="tokens_deployed_scan",
            processing_message_text="🔍 Analyzing tokens deployed by this wallet... This may take a moment.",
            error_message_text="❌ An error occurred while analyzing the tokens deployed. Please try again later.",
            no_data_message_text="❌ Could not find any tokens deployed by this wallet."
        )

    elif expecting == "track_wallet_buy_sell_address":
        wallet_address = update.message.text.strip()
        await handle_tracking_input(update, context, "wallet_trades", wallet_address)

    elif expecting == "track_new_token_deploy_address":
        wallet_address = update.message.text.strip()
        await handle_tracking_input(update, context, "token_deployments", wallet_address)

    elif expecting == "track_profitable_wallets_token":
        token_address = update.message.text.strip()
        await handle_tracking_input(update, context, "token_profitable_wallets", token_address)

    elif expecting == "kol_wallet_name":
        kol_wallet_name = update.message.text.strip()
        context.user_data["kol_wallet_name"] = kol_wallet_name
        
        keyboard = [
            [
                InlineKeyboardButton("1 Day", callback_data="kol_period_1"),
                InlineKeyboardButton("7 Days", callback_data="kol_period_7"),
                InlineKeyboardButton("30 Days", callback_data="kol_period_30")
            ],
            [InlineKeyboardButton("🔙 Back", callback_data="kol_wallets")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"📊 <b>KOL Wallet: {kol_wallet_name}</b>\n\n"
            f"Please select the time period for profitability analysis:",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )

# help handlers
async def handle_general_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle help button callback"""
    query = update.callback_query

    general_help_text = (
        "<b>🆘 Welcome to Crypto DeFi Analyze Help Center</b>\n\n"
        "I’m your trusted assistant for navigating the world of DeFi. Use me to analyze tokens, uncover wallet activity, and monitor top-performing or suspicious wallets across the blockchain. 🔍📈\n\n"
        
        "<b>📊 Token Analysis:</b>\n"
        "🔹 View the <b>first buyers</b> of any token (1-50 wallets) with full trade stats and PNL.\n"
        "🔹 Track the <b>All-Time High (ATH)</b> market cap of any token, and its current standing.\n"
        "🔹 Discover <b>most profitable wallets</b> holding a specific token.\n"
        "🔹 (Premium) Reveal the <b>deployer wallet</b> behind a token and their past projects.\n"
        "🔹 (Premium) Check <b>top holders</b> and whales, and track their activity.\n"
        "🔹 (Premium) Identify <b>High Net Worth wallets</b> with $10,000+ in token holdings.\n\n"
        
        "<b>🕵️ Wallet Analysis:</b>\n"
        "🔹 Analyze <b>wallet holding duration</b> – how long they hold tokens before selling.\n"
        "🔹 Discover <b>most profitable wallets</b> over 1 to 30 days.\n"
        "🔹 Find <b>top token deployer wallets</b> and their earnings.\n"
        "🔹 (Premium) View <b>all tokens deployed</b> by any wallet and their performance.\n\n"
        
        "<b>🔔 Tracking & Monitoring:</b>\n"
        "🔹 (Premium) <b>Track wallet buy/sell</b> actions in real-time.\n"
        "🔹 (Premium) Get alerts when a <b>wallet deploys new tokens</b> or is linked to new ones.\n"
        "🔹 (Premium) Analyze <b>profitable wallets</b> in any token across full metrics (PNL, trades, volume).\n\n"
        
        "<b>📢 KOL & Whale Monitoring:</b>\n"
        "🔹 Monitor <b>KOL wallets</b> and their profit/loss over time.\n"
        "🔹 (Premium) Get alerts when <b>top 10 holders or whales</b> buy or dump a token.\n\n"
        
        "<b>💎 Premium Access:</b>\n"
        "Unlock all features, unlimited scans, and powerful tracking with a Premium plan.\n\n"
        "Tap a button from the menu to start using a feature, or hit ⬅️ Back to return.\n"
        "Happy hunting in the DeFi jungle! 🌐🚀"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("📊 Token Analysis", callback_data="token_analysis_help"),
            InlineKeyboardButton("🕵️ Wallet Analysis", callback_data="wallet_analysis_help")
         ],
        [
            InlineKeyboardButton("🔔 Tracking & Monitoring",  callback_data="tracking_and_monitoring_help"),
            InlineKeyboardButton("🐳 KOL & Whale Monitoring", callback_data="kol_wallet_help")
        ],
        [InlineKeyboardButton("🔙 Back to Menu", callback_data="back")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    try:
        await query.edit_message_text(
            general_help_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
    )
    except Exception as e:
        logging.error(f"Error in handle_back: {e}")
        await query.message.reply_text(
            general_help_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        try:
            await query.message.delete()
        except:
            pass

async def handle_token_analysis_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle token analysis button callback"""
    query = update.callback_query
    
    token_analysis_help_text = (
        "<b>📊 TOKEN ANALYSIS HELP</b>\n\n"
        "Use these features to deeply analyze any token across the blockchain. 🔎📈\n\n"

        "<b>🏁 First Buyers & Profits</b>\n"
        "🔹 See the first 1-50 wallets that bought a token with full stats:\n"
        "   - Buy & sell amount, total trades, PNL, and win rate.\n"
        "   - (Free: 3 token scans/day, Premium: Unlimited)\n\n"

        "<b>💰 Most Profitable Wallets</b>\n"
        "🔹 Discover the most profitable wallets holding a specific token.\n"
        "   - Includes buy & sell totals and net profit.\n"
        "   - (Free: 3 token scans/day, Premium: Unlimited)\n\n"

        "<b>📈 Market Cap & ATH</b>\n"
        "🔹 View the all-time high (ATH) market cap of any token.\n"
        "   - Includes ATH date and % from ATH.\n"
        "   - (Free: 3 token scans/day, Premium: Unlimited)\n\n"

        "<b>🧠 Deployer Wallet Scan</b> (Premium)\n"
        "🔹 Reveal the deployer wallet and all tokens deployed by it.\n"
        "   - Includes ATH market cap and x-multipliers.\n\n"

        "<b>🐋 Top Holders & Whale Watch</b> (Premium)\n"
        "🔹 See the top 10 holders and whale wallets of a token.\n"
        "🔹 Get notified when Dev, whales, or top 10 holders sell.\n\n"

        "<b>💎 High Net Worth Wallets</b> (Premium)\n"
        "🔹 Scan for wallets holding over $10,000 worth of a token.\n"
        "   - Includes total worth in USD, token amount, and average holding time.\n"
    )

    keyboard = [
        [InlineKeyboardButton("🔙 Back to Menu", callback_data="token_analysis")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        await query.edit_message_text(
            token_analysis_help_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
    )
    except Exception as e:
        logging.error(f"Error in handle_back: {e}")
        await query.message.reply_text(
            token_analysis_help_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        try:
            await query.message.delete()
        except:
            pass

async def handle_wallet_analysis_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle help button callback"""
    query = update.callback_query
    
    wallet_analysis_help_text = (
        "<b>🕵️ WALLET ANALYSIS HELP</b>\n\n"
        "Analyze individual wallets to uncover trading behavior and profitability. 🧠📊\n\n"

        "<b>💎 Most Profitable Wallets (1–30 days)</b>\n"
        "🔹 Track wallets with highest profits in short timeframes.\n"
        "   - Includes total buy amount and trade count.\n"
        "   - (Free: 2 wallets, Premium: Unlimited)\n\n"

        "<b>🕒 Wallet Holding Duration</b>\n"
        "🔹 Check how long a wallet holds tokens before selling.\n"
        "   - (Free: 3 wallet scans/day, Premium: Unlimited)\n\n"

        "<b>🧪 Most Profitable Token Deployer Wallets</b>\n"
        "🔹 Find top-earning deployers in the last 1–30 days.\n"
        "   - (Free: 2 wallets, Premium: Unlimited)\n\n"

        "<b>🧱 Tokens Deployed by Wallet</b> (Premium)\n"
        "🔹 Scan a wallet to view tokens it deployed.\n"
        "   - Includes name, ticker, price, deployment date, market cap, ATH.\n"
    )

    keyboard = [
        [InlineKeyboardButton("🔙 Back to Menu", callback_data="wallet_analysis")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        await query.edit_message_text(
            wallet_analysis_help_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
    )
    except Exception as e:
        logging.error(f"Error in handle_back: {e}")

        await query.message.reply_text(
            wallet_analysis_help_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        
        try:
            await query.message.delete()
        except:
            pass

async def handle_tracking_and_monitoring_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle help button callback"""
    query = update.callback_query
    
    tracking_and_monitoring_help_text = (
    "<b>🔔 TRACKING & MONITORING HELP</b>\n\n"
    "Track wallets and token performance in real-time. Stay ahead of the game! ⚡👀\n\n"

    "<b>📈 Track Wallet Buy/Sell</b> (Premium)\n"
    "🔹 Get real-time alerts when a wallet buys or sells any token.\n\n"

    "<b>🧱 Track New Token Deployments</b> (Premium)\n"
    "🔹 Get notified when a wallet deploys a new token.\n"
    "🔹 Also alerts for new tokens linked to that wallet.\n\n"

    "<b>📊 Profitable Wallets of Any Token</b> (Premium)\n"
    "🔹 Track profitable wallets in any token.\n"
    "   - Full metrics: PNL, trades, volume, win rate (1–30 days).\n"
)

    keyboard = [
        [InlineKeyboardButton("🔙 Back to Menu", callback_data="tracking_and_monitoring")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        await query.edit_message_text(
            tracking_and_monitoring_help_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
    )
    except Exception as e:
        logging.error(f"Error in handle_back: {e}")

        await query.message.reply_text(
            tracking_and_monitoring_help_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        
        try:
            await query.message.delete()
        except:
            pass

async def handle_kol_wallets_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle help button callback"""
    query = update.callback_query
    
    kol_wallets_help_text = (
        "<b>📢 KOL & WHALE MONITORING HELP</b>\n\n"
        "Track influencers, devs, and whales in the crypto market. 🐳🧠\n\n"

        "<b>📊 KOL Wallets Profitability</b>\n"
        "🔹 Track influencer wallets' PNL over 1–30 days.\n"
        "   - (Free: 3 scans/day, Premium: Unlimited)\n\n"
    )

    keyboard = [
        [InlineKeyboardButton("🔙 Back to Menu", callback_data="kol_wallets")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        await query.edit_message_text(
            kol_wallets_help_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
    )
    except Exception as e:
        logging.error(f"Error in handle_back: {e}")

        await query.message.reply_text(
            kol_wallets_help_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        
        try:
            await query.message.delete()
        except:
            pass

# main menu handlers
async def handle_start_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the main menu display"""

    if "default_network" not in context.user_data:
        context.user_data["default_network"] = "eth"

    selected_network = context.user_data.get("default_network")
    network_display = {
        "eth": "🌐 Ethereum",
        "base": "🛡️ Base",
        "bsc": "🔶 BSC"
    }
    

    if not selected_network:
        await handle_select_network(update, context)
        return
    
    welcome_message = (
        f"🆘 <b>Welcome to Crypto DeFi Analyze Bot, {update.effective_user.first_name}! 🎉</b>\n\n"
        f"🔎 <b>Your DeFi Intelligence Hub</b>\n"
        f"Get ahead in crypto with powerful tools for wallet tracking, token analytics, and market insights. 📊💰\n\n"
        
        f"✨ <b>Core Features:</b>\n"
        f"• Token analysis, first buyers, PNL tracking, ATH market cap, deployer scan, top holders, high net worth wallets\n"
        f"• Wallet analysis, holding duration, top deployers, wallet-deployed tokens, profitable wallet history\n"
        f"• Real-time tracking for buys/sells, token deployments, and whale movements\n"
        f"• KOL tracking, influencer wallet profits, whale and dev sell alerts\n\n"
        
        f"🆓 Free users enjoy limited daily scans\n"
        f"💎 <b>Upgrade to Premium</b> for unlimited scans and full access to all features\n\n"

        f"Happy Trading! 🚀💰"
    )

    keyboard_main = [
        [
            InlineKeyboardButton("📊 Token Analysis", callback_data="token_analysis"),
            InlineKeyboardButton("🕵️ Wallet Analysis", callback_data="wallet_analysis"),
        ],
        [
            InlineKeyboardButton("🔔 Tracking & Monitoring", callback_data="tracking_and_monitoring"),
            InlineKeyboardButton("🐳 KOL wallets", callback_data="kol_wallets")
        ],
        [
            InlineKeyboardButton(f"🔗 Network ({network_display.get(selected_network, selected_network.upper())})", callback_data="select_network"),
            InlineKeyboardButton("❓ Help", callback_data="general_help"),
        ],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard_main)
    
    if update.callback_query:
        try:
            await update.callback_query.edit_message_text(
                welcome_message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logging.error(f"Error editing message in handle_start_menu: {e}")
            await update.callback_query.message.reply_text(
                welcome_message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
            try:
                await update.callback_query.message.delete()
            except:
                pass
    else:
        await update.message.reply_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )

async def handle_select_network(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle network selection from main menu"""
    query = update.callback_query

    current_network = context.user_data.get("default_network", "eth")  # Default to Ethereum if none set
    
    eth_label = "🌐 Ethereum (Current)" if current_network == "eth" else "🌐 Ethereum" 
    base_label = "🛡️ Base (Current)" if current_network == "base" else "🛡️ Base" 
    bsc_label = "🔶 BSC (Current)" if current_network == "bsc" else "🔶 BSC"
    
    keyboard = [
        [
            InlineKeyboardButton(eth_label, callback_data="set_default_network_eth"),
            InlineKeyboardButton(base_label, callback_data="set_default_network_base"),
            InlineKeyboardButton(bsc_label, callback_data="set_default_network_bsc"),
        ],
        [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text(
        "🔗 <b>Select Blockchain Network</b>\n\n"
        "Please choose the blockchain network you'd like to use for token and wallet analyses. Each chain has its own ecosystem, speed, and opportunities:\n\n"
        "• <b>🌐 Ethereum</b>: The original smart contract platform\n"
        "• <b>🛡️ Base</b>: Coinbase's Ethereum L2 solution\n"
        "• <b>🔶 BSC</b>: Binance Smart Chain (BNB Chain)\n\n"
        "This setting will be used for all token and wallet analyses.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )

async def handle_set_default_network(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle setting default network"""
    query = update.callback_query
    callback_data = query.data
    
    # Extract network from callback data
    # Format: "set_default_network_{network}"
    network = callback_data.replace("set_default_network_", "")
    
    # Map of network to display name
    network_display = {
        "eth": "🌐 Ethereum",
        "base": "🛡️ Base",
        "bsc": "🔶 BSC"
    }
    
    # Store the selected network in user_data
    context.user_data["default_network"] = network
    
    # Confirm selection
    await query.answer(f"Network set to {network_display.get(network, network.upper())}")
    
    # Return to main menu with the selected network
    await handle_start_menu(update, context)

async def handle_token_analysis(update:Update, context:ContextTypes.DEFAULT_TYPE)->None: 
    """Handle token analysis button"""

    if not context.user_data.get("default_network"):
        await handle_select_network(update, context)
        return

    welcome_message = (
        f"✨ Welcome to <b>📊 Token Analysis:</b>\n\n"
        f"🔹 <b>First Buyers & Profits of a token:</b> See the first 1-50 buy wallets of a token with buy & sell amount, buy & sell trades, total trades and PNL and win rate. (Maximum 3 token scans daily only for free users. Unlimited token scans daily for premium users)\n\n"
        f"🔹 <b>Most Profitable Wallets of a token:</b> Most profitable wallets in any specific token with total buy & sell amount and profit. (Maximum 3 token scans daily only for free users. Unlimited token scans daily for premium users)\n\n"
        f"🔹 <b>Market Cap & ATH:</b>All time high (ATH) market cap of any token with date and percentage of current market cap from ATH marketcap. (Maximum 3 token scans daily only for free users. Unlimited token scans daily for premium users)\n\n"
        f"🔹 <b>Deployer Wallet Scan:</b> (Premium) Scan a token contract to reveal the deployer wallet and show other tokens ever deployed by the deployer wallet and their all time high (ATH) marketcap and how many X's they did.\n\n"
        f"🔹 <b>Top Holders & Whale Watch:</b> (Premium) Scan a token contract to see top 10 holders, whale wallets holding the token.\n\n"
        f"🔹 <b>High Net Worth Wallet Holders:</b> (Premium) High net worth wallet holders of any token with total worth of at least $10,000 showing total worth in USD, coins/tokens held and amount and average holding time of the wallet.\n\n"
        f"🔹 <b>💎 Upgrade to Premium:</b> Unlock unlimited scans and premium features.\n\n"
        f"Happy Trading! 🚀💰"
    )

    token_analysis_keyboard = [
        [InlineKeyboardButton("🛒 First Buyers & Profits of a token", callback_data="token_first_buyers")],
        [InlineKeyboardButton("💰 Most Profitable Wallets of a token", callback_data="token_most_profitable_wallets")],
        [InlineKeyboardButton("📈 Market Cap & ATH of a token", callback_data="token_ath")],
        [InlineKeyboardButton("🧑‍💻 Deployer Wallet Scan (Premium)", callback_data="token_deployer_wallet_scan")],
        [InlineKeyboardButton("🐳 Top Holders & Whale Watch (Premium)", callback_data="token_top_holders")],
        [InlineKeyboardButton("💼 High Net Worth Holders (Premium)", callback_data="token_high_net_worth_holders")],
        [
            InlineKeyboardButton("❓ Help", callback_data="token_analysis_help"),
            InlineKeyboardButton("🔙 Back", callback_data="back")
        ],
    ]
    
    reply_markup = InlineKeyboardMarkup(token_analysis_keyboard)
    
    # Check if this is a callback query or a direct message
    if update.callback_query:
        await update.callback_query.edit_message_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    else:
        await update.message.reply_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )

async def handle_wallet_analysis(update:Update, context:ContextTypes.DEFAULT_TYPE)->None: 
    """Handle wallet analysis button"""
    
    welcome_message = (
        f"✨ Welcome to <b>🕵️ Wallet Analysis:</b>\n\n"
        f"🔹 <b>Most profitable wallets in a specific period:</b>Most profitable wallets in 1 to 30 days with total buy amount and number of trades. (Free users get only 2 most profitable wallets from this query. Premium users get unlimited)\n\n"
        f"🔹 <b>Wallet Holding Duration:</b> See how long a wallet holds a token before selling. (Maximum 3 wallet scans daily only for free users. Unlimited wallet scans daily for premium users)\n\n"
        f"🔹 <b>Most profitable token deployer wallets:</b> See the most profitable token deployer wallets in 1 to 30 days. (Free users only get 2 most profitable token deployer wallets from this query. Premium users get unlimited)\n\n"
        f"🔹 <b>Tokens Deployed by Wallet:</b> (Premium) See the tokens deployed by a particular wallet showing token name, ticker/symbol, current price, date of deployment, current market cap and All Time High (ATH) market cap.\n\n"
        f"Happy Trading! 🚀💰"
    )
    wallet_tracking_keyboard = [
        [InlineKeyboardButton("💹 Most profitable wallets in specific period", callback_data="wallet_most_profitable_in_period")],
        [InlineKeyboardButton("💰 Most profitable token deployer wallets in period", callback_data="most_profitable_token_deployer_wallet")],
        [InlineKeyboardButton("⏳ Wallet Holding Duration", callback_data="wallet_holding_duration")],
        [InlineKeyboardButton("🚀 Tokens Deployed by Wallet (Premium)", callback_data="tokens_deployed_by_wallet")],
        [
            InlineKeyboardButton("❓ Help", callback_data="wallet_analysis_help"),
            InlineKeyboardButton("🔙 Back", callback_data="back")
        ],
    ]
    
    reply_markup = InlineKeyboardMarkup(wallet_tracking_keyboard)
    
    # Check if this is a callback query or a direct message
    if update.callback_query:
        await update.callback_query.edit_message_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    else:
        await update.message.reply_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )

async def handle_tracking_and_monitoring(update:Update, context:ContextTypes.DEFAULT_TYPE)->None: 
    """Handle tracking and monitoring button"""
    welcome_message = (
        f"✨ Welcome to <b>🔔 Tracking & Monitoring</b>\n\n"
        f"🔹 <b>Track Buy/Sell Activity:</b> (Premium) Track a wallet to be notified when the wallet buys or sells any token.\n\n"
        f"🔹 <b>Track New Token Deployments:</b> (Premium) Track a wallet to be notified when that wallet deploys a new token or any of the wallet it's connected to deploys a new token.\n\n"
        f"🔹 <b>Profitable Wallets of any token:</b> (Premium) Track the profitable wallets in any token with total maximum number of trades, PNL, buy amount, sell amount, buy volume, sell volume, and win rate within 1 to 30 days.\n\n"
        f"Happy Trading! 🚀💰"
    )

    tracking_and_monitoring_keyboard = [
        [InlineKeyboardButton("📥 Track Buy/Sell Activity (Premium)", callback_data="track_wallet_buy_sell")],
        [InlineKeyboardButton("🧬 Track Token Deployments (Premium)", callback_data="track_new_token_deploy")],
        [InlineKeyboardButton("📊 Profitable Wallets of a token(Premium)", callback_data="track_profitable_wallets")],
        [InlineKeyboardButton("👁️ View Active Tracking", callback_data="view_tracking_subscriptions")],
        [
            InlineKeyboardButton("❓ Help", callback_data="tracking_and_monitoring_help"),
            InlineKeyboardButton("🔙 Back", callback_data="back")
        ],
    ]
    
    reply_markup = InlineKeyboardMarkup(tracking_and_monitoring_keyboard)
    
    # Check if this is a callback query or a direct message
    if update.callback_query:
        await update.callback_query.edit_message_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    else:
        await update.message.reply_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )

async def handle_kol_wallets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle kol wallets button"""
    welcome_message = (
        f"✨ Welcome to <b>🐳 KOL wallets</b>\n\n"
        f"🔹 <b>KOL Wallets Profitability:</b> Track KOL wallets profitability in 1-30 days with wallet name and PNL. (Maximum 3 scans daily only for free users. Unlimited scans daily for premium users)\n\n"
        f"Happy Trading! 🚀💰"
    )
    token_analysis_keyboard = [
        [InlineKeyboardButton("📢 KOL Wallets Profitability", callback_data="kol_wallet_profitability")],
        [
            InlineKeyboardButton("❓ Help", callback_data="kol_wallets_help"),
            InlineKeyboardButton("🔙 Back", callback_data="back")
        ],
    ]
    
    reply_markup = InlineKeyboardMarkup(token_analysis_keyboard)
    
    # Check if this is a callback query or a direct message
    if update.callback_query:
        await update.callback_query.edit_message_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    else:
        await update.message.reply_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )

async def handle_setup_whale_tracking(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle setup whale tracking callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    callback_data = query.data
    token_address = callback_data.replace("setup_whale_tracking_", "")
    
    await query.answer("Setting up tracking...")
    processing_message = await query.message.reply_text(
        "🔄 Setting up whale and top holder tracking... This may take a moment."
    )
    
    try:
        chain=context.user_data.get("default_network", "eth")
        token_info = await get_token_info(token_address)
        
        top_holders = await get_token_top_holders(token_address, chain)
        
        # Get deployer wallet
        deployer_info = await get_deployer_wallet_scan_data(token_address, chain)
        deployer_wallet = deployer_info.get("deployer_address") if deployer_info else None
        
        # Create token tracking subscription with metadata
        token_subscription = TrackingSubscription(
            user_id=user.user_id,
            tracking_type="token_whale_tracking",
            target_address=token_address,
            is_active=True,
            created_at=datetime.now(),
            metadata={
                "token_name": token_info.get("name", "Unknown"),
                "token_symbol": token_info.get("symbol", "Unknown"),
                "deployer": deployer_wallet
            }
        )
        save_tracking_subscription(token_subscription)
        
        # Track deployer wallet if available
        if deployer_wallet:
            deployer_subscription = TrackingSubscription(
                user_id=user.user_id,
                tracking_type="wallet_trades",
                target_address=deployer_wallet,
                is_active=True,
                created_at=datetime.now(),
                metadata={
                    "role": "deployer",
                    "token": token_address,
                    "token_name": token_info.get("name", "Unknown"),
                    "token_symbol": token_info.get("symbol", "Unknown")
                }
            )
            save_tracking_subscription(deployer_subscription)
        
        # Track top holders
        for holder in top_holders:
            holder_subscription = TrackingSubscription(
                user_id=user.user_id,
                tracking_type="wallet_trades",
                target_address=holder["address"],
                is_active=True,
                created_at=datetime.now(),
                metadata={
                    "role": "top_holder",
                    "token": token_address,
                    "token_name": token_info.get("name", "Unknown"),
                    "token_symbol": token_info.get("symbol", "Unknown"),
                    "percentage": holder.get("percentage", 0)
                }
            )
            save_tracking_subscription(holder_subscription)
        
        # Format confirmation message
        response = (
            f"✅ <b>Whale Tracking Set Up Successfully!</b>\n\n"
            f"Now tracking whales and top holders for token:\n"
            f"<b>{token_info.get('name', 'Unknown')}</b> ({token_info.get('symbol', 'N/A')})\n"
            f"`{token_address[:6]}...{token_address[-4:]}`\n\n"
        )
        
        if deployer_wallet:
            response += (
                f"<b>🧑‍💻 Deployer Wallet:</b>\n"
                f"`{deployer_wallet[:6]}...{deployer_wallet[-4:]}`\n\n"
            )
        
        response += "<b>🐳 Top Holders Being Tracked:</b>\n"
        
        for i, holder in enumerate(top_holders[:5], 1):  # Show first 5 for brevity
            percentage = holder.get('percentage', 0)
            response += (
                f"{i}. `{holder['address'][:6]}...{holder['address'][-4:]}`\n"
                f"   Holdings: {percentage:.2f}% of supply\n"
            )
        
        if len(top_holders) > 5:
            response += f"\n...and {len(top_holders) - 5} more holders\n"
        
        response += (
            f"\n<b>🚨 You will receive alerts under the following conditions:</b>\n"
            f"• When the project’s developer decides to sell a portion or all of their tokens, indicating a potential shift in confidence or strategy.\n"
            f"• When any of the top 10 largest holders initiate a sell-off, which could signal major market movements or sentiment changes.\n"
            f"• When a known whale wallet — a large and influential holder — begins dumping the token in significant volumes, which might lead to volatility or price drops."
        )
        
        # Add button to view all tracking subscriptions
        keyboard = [
            [InlineKeyboardButton("👁️ View All Tracking", callback_data="view_tracking_subscriptions")],
            [InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await processing_message.edit_text(
            response,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        
    except Exception as e:
        logging.error(f"Error in handle_setup_whale_tracking: {e}")
        await processing_message.edit_text(
            "❌ An error occurred while setting up whale tracking. Please try again later."
        )


# token analysis handlers
async def handle_first_buyers(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle first buyers button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Check if user has reached daily limit
    has_reached_limit, current_count = await check_rate_limit_service(
        user.user_id, "first_buy_wallet_scan", FREE_TOKEN_SCANS_DAILY
    )
    
    if has_reached_limit and not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            f"⚠️ <b>Daily Limit Reached</b>\n\n"
            f"🧾 You've used <b>{current_count}</b> out of <b>{FREE_TOKEN_SCANS_DAILY}</b> free daily scans for <b>First Buyers Analysis</b>.\n"
            f"This feature lets you discover who bought early, how much they earned, and their trading behavior. Great for identifying smart money moves! 💸\n\n"
            f"💎 <b>Upgrade to Premium</b> for unlimited scans and deeper DeFi intelligence:\n",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    await handle_token_analysis_token_input(update, context, "first_buyers")

async def handle_token_most_profitable_wallets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle most profitable wallets button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Check if user has reached daily limit
    has_reached_limit, current_count = await check_rate_limit_service(
        user.user_id, "token_most_profitable_wallet_scan", FREE_TOKEN_SCANS_DAILY
    )
    
    if has_reached_limit and not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            f"⚠️ <b>Daily Limit Reached</b>\n\n"
            f"📊 You've used <b>{current_count}</b> out of <b>{FREE_TOKEN_SCANS_DAILY}</b> daily scans for <b>Most Profitable Wallets</b>.\n"
            f"This feature helps you uncover top-performing wallets in any token — who's buying, who's profiting, and how much! 🧠💰\n\n"
            f"💎 <b>Premium users enjoy unlimited scans</b> and access to full profitability metrics:\n",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    # Prompt user to select a chain
    await handle_token_analysis_token_input(update, context, "token_most_profitable_wallets")

async def handle_ath(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle ATH button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Check if user has reached daily limit
    has_reached_limit, current_count = await check_rate_limit_service(
        user.user_id, "ath_scan", FREE_TOKEN_SCANS_DAILY
    )
    
    if has_reached_limit and not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            f"⚠️ <b>Daily Limit Reached</b>\n\n"
            f"🚫 You've used <b>{current_count}</b> out of <b>{FREE_TOKEN_SCANS_DAILY}</b> free daily token scans.\n"
            f"Free users can analyze up to {FREE_TOKEN_SCANS_DAILY} tokens each day to explore market caps, trends, and ATH insights.\n\n"
            f"💎 <b>Premium users get unlimited scans</b> — no restrictions, no waiting!\n"
            f"🚀 <b>Upgrade now and dive deeper into DeFi intelligence!</b>",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    # Prompt user to select a chain
    await handle_token_analysis_token_input(update, context, "ath")

async def handle_deployer_wallet_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle deployer wallet scan button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Check if user is premium
    if not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            "⭐ <b>Premium Feature</b>\n\n"
            "🔐 <b>Deployer Wallet Scanning</b> is an advanced feature available only to <b>Premium</b> users.\n"
            "This feature shows the deployer wallet, tokens they've launched, ATH market caps, and how many X’s they did — perfect for spotting trends and smart deployers early.\n\n"
            "💎 <b>Upgrade to Premium</b> now to unlock this and many more pro features!",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
        
    # Prompt user to select a chain
    await handle_token_analysis_token_input(update, context, "deployer_wallet_scan")

async def handle_top_holders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle top holders button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Check if user is premium
    if not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            "⭐ <b>Premium Feature</b>\n\n"
            "🔍 <b>Top Holders & Whale Analysis</b> is an advanced feature available only to <b>Premium</b> users.\n" 
            "💎 <b>Upgrade to Premium</b> for full access to whale tracking, token alerts, unlimited scans, and deeper DeFi insights — stay ahead of the game! 🚀",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    # Prompt user to select a chain
    await handle_token_analysis_token_input(update, context, "top_holders")

async def handle_high_net_worth_holders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle high net worth token holders button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Check if user is premium
    if not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            "⭐ <b>Premium Feature</b>\n\n"
            "💰 <b>High Net Worth Holders</b> analysis is a powerful tool exclusive to <b>Premium</b> users. It reveals wallets holding over $10K+, with insights into token value, quantities held, and average holding time.\n\n"
            "💎 <b>Upgrade to Premium</b> for full access to whale tracking, token alerts, unlimited scans, and deeper DeFi insights — stay ahead of the game! 🚀",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )

        return
    
    # Prompt user to select a chain
    await handle_token_analysis_token_input(update, context, "high_net_worth_holders")

async def handle_token_analysis_token_input(update: Update, context: ContextTypes.DEFAULT_TYPE, feature: str) -> None:
    """
    Generic function to prompt user to select a blockchain network
    
    Args:
        update: The update object
        context: The context object
        feature: The feature identifier (e.g., 'first_buyers', 'ath', etc.)
    """
    query = update.callback_query
    chain = context.user_data.get("default_network")
        
    # Map of feature to expecting state and display name
    feature_map = {
        "first_buyers": {
            "expecting": "first_buyers_token_address",
            "display": "first buyers"
        },
        "token_most_profitable_wallets": {
            "expecting": "token_most_profitable_wallets_token_address",
            "display": "most profitable wallets"
        },
        "ath": {
            "expecting": "ath_token_address",
            "display": "ATH data"
        },
        "deployer_wallet_scan": {
            "expecting": "deployer_wallet_scan_token",
            "display": "deployer wallet"
        },
        "top_holders": {
            "expecting": "top_holders_token_address",
            "display": "top holders"
        },
        "high_net_worth_holders": {
            "expecting": "high_net_worth_holders_token_address",
            "display": "high net worth holders"
        }
    }

    # Map of chain to display name
    chain_display = {
        "eth": "🌐 Ethereum",
        "base": "🛡️ Base",
        "bsc": "🔶 BSC"
    }
    
    # Get feature info
    feature_info = feature_map.get(feature, {"expecting": "unknown", "display": feature})
    
    # Prompt user to enter token address with back button
    keyboard = [
        [InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text(
        f"🔍 <b>Token Analysis on {chain_display.get(chain, chain.upper())}</b>\n\n"
        f"Please send me the token contract address to analyze its {feature_info['display']}.\n\n"
        f"Example: `0x1234...abcd`",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )
    
    # Set conversation state to expect token address for the specific feature
    context.user_data["expecting"] = feature_info["expecting"]

async def handle_chain_selection_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle chain selection callbacks"""
    query = update.callback_query
    callback_data = query.data
    
    # Extract feature and chain from callback data
    # Format: "{feature}_chain_{chain}"
    parts = callback_data.split("_chain_")
    if len(parts) != 2:
        await query.answer("Invalid selection", show_alert=True)
        return
    
    feature = parts[0]
    chain = parts[1]
    
    # Store the selected chain in user_data
    context.user_data["selected_chain"] = chain
    
    # Map of feature to expecting state and display name
    feature_map = {
        "first_buyers": {
            "expecting": "first_buyers_token_address",
            "display": "first buyers"
        },
        "token_most_profitable_wallets": {
            "expecting": "token_most_profitable_wallets_token_address",
            "display": "most profitable wallets"
        },
        "ath": {
            "expecting": "ath_token_address",
            "display": "ATH data"
        },
        "deployer_wallet_scan": {
            "expecting": "deployer_wallet_scan_token",
            "display": "deployer wallet"
        },
        "top_holders": {
            "expecting": "top_holders_token_address",
            "display": "top holders"
        },
        "high_net_worth_holders": {
            "expecting": "high_net_worth_holders_token_address",
            "display": "high net worth holders"
        }
    }
    
    # Map of chain to display name
    chain_display = {
        "eth": "🌐 Ethereum",
        "base": "🛡️ Base",
        "bsc": "🔶 BSC"
    }
    
    # Get feature info
    feature_info = feature_map.get(feature, {"expecting": "unknown", "display": feature})
    
    # Prompt user to enter token address with back button
    keyboard = [
        [InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        f"🔍 <b>Token Analysis on {chain_display.get(chain, chain.upper())}</b>\n\n"
        f"Please send me the token contract address to analyze its {feature_info['display']}.\n\n"
        f"Example: `0x1234...abcd`",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )
    
    # Set conversation state to expect token address for the specific feature
    context.user_data["expecting"] = feature_info["expecting"]

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
        get_data_func: Function to get the data (takes token_address and chain)
        format_response_func: Function to format the response (takes data and token_data)
        scan_count_type: Type of scan to increment count for
        processing_message_text: Text to show while processing
        error_message_text: Text to show on error
        no_data_message_text: Text to show when no data is found
    """
    token_address = update.message.text.strip()
    selected_chain = context.user_data.get("default_network")
    
    # Validate address
    if not await is_valid_token_contract(token_address, selected_chain):
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"⚠️ Something went wrong.⚠️ Please provide a valid token contract address for {selected_chain}.",
            reply_markup=reply_markup
        )
        return
    
    # Send processing message
    processing_message = await update.message.reply_text(processing_message_text)
    
    try:
        # Get data
        token_info = await get_token_info(token_address, selected_chain)
        data = await get_data_func(token_address, selected_chain)
        
        if not data or not token_info:
            # Add back button when no data is found
            keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="token_analysis")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await processing_message.edit_text(
                no_data_message_text,
                reply_markup=reply_markup
            )
            return
        
        # Format the response
        response, keyboard = format_response_func(data, token_info, token_address)

        if analysis_type == "top_holders":
            keyboard.insert(0,[InlineKeyboardButton("🔔 Track Whale & Top Holder Sells", callback_data=f"setup_whale_tracking_{token_address}")])
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


# wallet analysis handlers
async def handle_wallet_analysis_input(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    analysis_type: str,
    get_data_func,
    format_response_func,
    scan_count_type: str,
    processing_message_text: str,
    error_message_text: str,
    no_data_message_text: str,
) -> None:
    """
    Generic handler for wallet analysis inputs
    
    Args:
        update: The update object
        context: The context object
        analysis_type: Type of analysis being performed (for logging)
        get_data_func: Function to get the data (takes wallet_address)
        format_response_func: Function to format the response (takes data)
        scan_count_type: Type of scan to increment count for
        processing_message_text: Text to show while processing
        error_message_text: Text to show on error
        no_data_message_text: Text to show when no data is found
        additional_params: Additional parameters to pass to get_data_func
    """
    wallet_address = update.message.text.strip()
    selected_chain = context.user_data.get("selected_chain", "eth")
    
    if not await is_valid_wallet_address(wallet_address, selected_chain):
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="wallet_analysis")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
               
        await update.message.reply_text(
            f"⚠️ Something went wrong.⚠️ Please provide a valid wallet address on {selected_chain}.",
            reply_markup=reply_markup
        )
        return
    
    # Send processing message
    processing_message = await update.message.reply_text(processing_message_text)
    try:
        # Get data
        data = await get_data_func(wallet_address, selected_chain)
        
        if not data:
            # Add back button when no data is found
            keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="wallet_analysis")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await processing_message.edit_text(
                no_data_message_text,
                reply_markup=reply_markup
            )
            return
        
        # Format the response
        response, keyboard = format_response_func(data, wallet_address)
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
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="wallet_analysis")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await processing_message.edit_text(
            error_message_text,
            reply_markup=reply_markup
        )

async def handle_wallet_most_profitable_in_period(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle most profitable wallets in period button callback"""
    
    await handle_period_selection(
        update=update,
        context=context,
        feature_info="Most Profitable Wallets Analysis",
        scan_type="wallet_most_profitable_in_period_scan",
        callback_prefix="profitable_period"
    )

async def handle_most_profitable_token_deployer_wallet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await handle_period_selection(
        update=update,
        context=context,
        feature_info="Most Profitable Token Deployer Wallets Analysis",
        scan_type="most_profitable_token_depolyer_wallet_in_period_scan",
        callback_prefix="deployer_period"
    )
    
async def handle_wallet_holding_duration(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle wallet holding duration button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Check if user has reached daily limit
    has_reached_limit, current_count = await check_rate_limit_service(
        user.user_id, "wallet_holding_duration_scan", FREE_WALLET_SCANS_DAILY
    )
    
    if has_reached_limit and not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back", callback_data="wallet_analysis")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.message.reply_text(
            f"⚠️ <b>Daily Limit Reached</b>\n\n"
            f"🧾 You've used <b>{current_count}</b> out of <b>{FREE_WALLET_SCANS_DAILY}</b> free daily scans for <b>Wallet Intelligence Analysis</b>.\n"
            f"This feature reveals wallet behavior, holding duration, token deployment patterns, and profitability over time. Perfect for spotting smart wallets and high-performing traders! 🧠📊\n\n"
            f"💎 <b>Upgrade to Premium</b> for unlimited access and advanced features:\n"
            f"• Run unlimited wallet scans 🔍\n"
            f"• Uncover hidden whales and top deployers 🐋\n"
            f"• Get profit breakdowns, claim behavior, and token interaction data 📈\n"
            f"• Dive deep into DeFi with powerful analytics 🚀\n\n"
            f"🔓 Unlock deeper on-chain intelligence with Premium today!",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )

        return
    
    await handle_wallet_analysis_wallet_input(update, context, "wallet_holding_duration")

async def handle_tokens_deployed_by_wallet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle tokens deployed by wallet button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Check if user is premium
    if not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back", callback_data="wallet_analysis")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            "⭐ <b>Premium Feature</b>\n\n"
            "🔍 <b>Tokens Deployed by Wallet</b> is a powerful analytics tool that reveals every token a wallet has deployed—great for identifying smart contract creators, tracking deployer activity, and spotting patterns early! 🧠💥\n\n"
            "🚫 This feature is currently available only for <b>Premium users</b>.\n\n"
            "💎 <b>Upgrade to Premium</b> and unlock full access to:\n"
            "• Token deployment histories 📜\n"
            "• Wallet profit and claim tracking 💰\n"
            "• Early buyer detection tools 📈\n"
            "• Deep dive DeFi intelligence 🚀\n\n"
            "🔓 Tap into the full power of Crypto DeFi Analyze with Premium access!",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    # Prompt user to input wallet address
    await handle_wallet_analysis_wallet_input(update, context, "tokens_deployed_by_wallet")

async def handle_wallet_analysis_wallet_input(update: Update, context: ContextTypes.DEFAULT_TYPE, feature: str) -> None:
    """
    Generic function to prompt user to input a wallet address for analysis
    
    Args:
        update: The update object
        context: The context object
        feature: The feature identifier (e.g., 'wallet_holding_duration', 'tokens_deployed_by_wallet', etc.)
    """
    query = update.callback_query
    chain = context.user_data.get("default_network", "eth")
    
    # Map of feature to expecting state and display name
    feature_map = {
        "wallet_holding_duration": {
            "expecting": "wallet_holding_duration_address",
            "display": "token holding duration"
        },
        "tokens_deployed_by_wallet": {
            "expecting": "tokens_deployed_wallet_address",
            "display": "tokens deployed"
        }
    }

    # Map of chain to display name
    chain_display = {
        "eth": "🌐 Ethereum",
        "base": "🛡️ Base",
        "bsc": "🔶 BSC"
    }
    
    # Get feature info
    feature_info = feature_map.get(feature, {"expecting": "unknown", "display": feature})
    
    # Prompt user to enter wallet address with back button
    keyboard = [
        [InlineKeyboardButton("🔙 Back", callback_data="wallet_analysis")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text(
        f"🔍 <b>Wallet Analysis on {chain_display.get(chain, chain.upper())}</b>\n\n"
        f"Please send me the wallet address to analyze its {feature_info['display']}.\n\n"
        f"Example: `0x1234...abcd`",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )
    
    # Set conversation state to expect wallet address for the specific feature
    context.user_data["expecting"] = feature_info["expecting"]



async def handle_period_selection_callback(
    update: Update, 
    context: ContextTypes.DEFAULT_TYPE,
    get_data_func,
    format_response_func,
    scan_count_type: str,
    processing_message_text: str,
    error_message_text: str,
    no_data_message_text: str,
    free_limit: int = FREE_RESPONSE_DAILY,
    premium_limit: int = 10
) -> None:
    """
    Generic handler for period selection callbacks
    
    Args:
        update: The update object
        context: The context object
        get_data_func: Function to get the data
        format_response_func: Function to format the response
        scan_count_type: Type of scan to increment count for
        processing_message_text: Text to show while processing
        error_message_text: Text to show on error
        no_data_message_text: Text to show when no data is found
        free_limit: Limit for free users
        premium_limit: Limit for premium users
    """
    query = update.callback_query
    user = await check_callback_user(update)
    
    selected_period = int(query.data.split("_")[-1])
    logging.info(f"Selected period: {selected_period} days")
    
    # Store the selected period in context
    context.user_data["selected_period"] = selected_period
    
    # Get the selected chain
    selected_chain = context.user_data.get("selected_chain", "eth")
    logging.info(f"Selected chain: {selected_chain}")    
    # Process the request
    processing_message = await query.edit_message_text(
        processing_message_text.format(days=selected_period)
    )
    
    try:
        # For free users, limit the number of results
        limit = premium_limit if user.is_premium else free_limit
        logging.info(f"User is premium: {user.is_premium}, using limit: {limit}")
        logging.info(f"Calling get_data_func with days={selected_period}, limit={limit}, chain={selected_chain}")

        data = await get_data_func(
            days=selected_period,
            limit=limit,
            chain=selected_chain
        )
        
        if not data:
            if get_data_func.__name__ == "get_kol_data_with_name":
                keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="kol_wallets")]]

            else:
                keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="wallet_analysis")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await processing_message.edit_text(
                no_data_message_text,
                reply_markup=reply_markup
            )
            return
        
        # Format the response
        response, keyboard = format_response_func(data)
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await processing_message.edit_text(
            response,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        
        # Increment scan count
        await increment_scan_count(user.user_id, scan_count_type)
        
    except Exception as e:
        logging.error(f"Error in handle_period_selection_callback: {e}")
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="wallet_analysis")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await processing_message.edit_text(
            error_message_text,
            reply_markup=reply_markup
        )

async def handle_profitable_period_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle selection of time period for profitable wallets analysis"""
    logging.info("Welcome to handle_profitable_period_selection_callback")
    
    await handle_period_selection_callback(
        update=update,
        context=context,
        scan_count_type="wallet_most_profitable_in_period_scan",
        get_data_func=get_wallet_most_profitable_in_period,
        format_response_func=format_wallet_most_profitable_response,
        processing_message_text="🔍 Finding most profitable wallets in the last {days} days... This may take a moment.",
        error_message_text="❌ An error occurred while analyzing profitable wallets. Please try again later.",
        no_data_message_text="❌ Could not find profitable wallets for this period.",
        free_limit=FREE_RESPONSE_DAILY,
        premium_limit=PREMIUM_RESPONSE_DAILY
    )

async def handle_deployer_period_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle selection of time period for profitable token deployer wallets analysis"""
    
    await handle_period_selection_callback(
        update=update,
        context=context,
        get_data_func=get_most_profitable_token_deployer_wallets,
        format_response_func=format_deployer_wallets_response,
        scan_count_type="most_profitable_token_deployer_scan",
        processing_message_text="🔍 Finding most profitable token deployers in the last {days} days... This may take a moment.",
        error_message_text="❌ An error occurred while analyzing profitable token deployers. Please try again later.",
        no_data_message_text="❌ Could not find profitable token deployers for this period.",
        free_limit=FREE_RESPONSE_DAILY,
        premium_limit=PREMIUM_RESPONSE_DAILY
    )

# track and monitoring handlers
async def handle_track_wallet_buy_sell(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle track wallet buy/sell button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Check if user is premium
    if not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back", callback_data="tracking_and_monitoring")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            "⭐ <b>Premium Feature</b>\n\n"
            "🔍 <b>Track Wallet Buy/Sell Activity</b> is a powerful monitoring tool that alerts you when a specific wallet makes trades. Perfect for following smart money, tracking whales, or monitoring suspicious wallets! 🕵️‍♂️💰\n\n"
            "🚫 This feature is currently available only for <b>Premium users</b>.\n\n"
            "💎 <b>Upgrade to Premium</b> and unlock full access to:\n"
            "• Real-time buy/sell notifications 📲\n"
            "• Track multiple wallets simultaneously 📊\n"
            "• Get token amount and value details 💵\n"
            "• Stay ahead of market movers 🚀\n\n"
            "🔓 Tap into the full power of Crypto DeFi Analyze with Premium access!",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="tracking_and_monitoring")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Prompt user to enter wallet address
    await query.message.reply_text(
        "Please send me the wallet address you want to track for buy/sell activities.\n\n"
        "Example: `0x1234...abcd`\n\n"
        "You'll receive notifications when this wallet makes significant trades.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )
    
    # Set conversation state to expect wallet address for tracking
    context.user_data["expecting"] = "track_wallet_buy_sell_address"
    # await handle_token_analysis_token_input(update, context, "track_wallet_buy_sell")

async def handle_track_new_token_deploy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle track new token deployments button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Check if user is premium
    if not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back", callback_data="tracking_and_monitoring")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            "⭐ <b>Premium Feature</b>\n\n"
            "🧬 <b>Track Token Deployments</b> lets you monitor when a specific wallet deploys new tokens or connects to newly deployed contracts. Essential for following prolific developers, tracking project teams, or getting early on new launches! 🚀🔭\n\n"
            "🚫 This feature is currently available only for <b>Premium users</b>.\n\n"
            "💎 <b>Upgrade to Premium</b> and unlock full access to:\n"
            "• Instant new token deployment alerts 🔔\n"
            "• Track multiple deployer wallets 👥\n"
            "• Get token contract details and initial parameters 📝\n"
            "• Be first to discover new projects 🥇\n\n"
            "🔓 Tap into the full power of Crypto DeFi Analyze with Premium access!",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="tracking_and_monitoring")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Prompt user to enter wallet address
    await query.message.reply_text(
        "🧬 <b>Track Token Deployments</b>\n\n"
        "Please send me the wallet address you want to track for new token deployments.\n\n"
        "Example: `0x1234...abcd`\n\n"
        "You'll receive notifications when this wallet deploys new tokens or connects to newly deployed contracts.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )
    
    # Set conversation state to expect wallet address for tracking
    context.user_data["expecting"] = "track_new_token_deploy_address"

async def handle_track_profitable_wallets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle track profitable wallets button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Check if user is premium
    if not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back", callback_data="tracking_and_monitoring")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            "⭐ <b>Premium Feature</b>\n\n"
            "📊 <b>Track Profitable Wallets</b> identifies and monitors the most successful wallets trading a specific token. Perfect for finding winning strategies, spotting smart money, and learning from top traders! 💸📈\n\n"
            "🚫 This feature is currently available only for <b>Premium users</b>.\n\n"
            "💎 <b>Upgrade to Premium</b> and unlock full access to:\n"
            "• Detailed PNL and win rate metrics 📊\n"
            "• Trade volume and position size analysis 💰\n"
            "• Track multiple tokens' profitable wallets 🔄\n"
            "• Get alerts on significant profitable wallet movements 🚨\n\n"
            "🔓 Tap into the full power of Crypto DeFi Analyze with Premium access!",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="tracking_and_monitoring")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Prompt user to enter token address
    await query.message.reply_text(
        "📊 <b>Track Profitable Wallets</b>\n\n"
        "Please send me the token contract address to track its most profitable wallets.\n\n"
        "Example: `0x1234...abcd`\n\n"
        "You'll receive detailed analysis of the most profitable wallets trading this token.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )
    
    # Set conversation state to expect token address for tracking
    context.user_data["expecting"] = "track_profitable_wallets_token"


async def handle_tracking_input(update: Update, context: ContextTypes.DEFAULT_TYPE, tracking_type: str, target_address: str) -> None:
    """
    Handle tracking input from users
    
    Args:
        update: The update object
        context: The context object
        tracking_type: Type of tracking (wallet_trades, token_deployments, token_profitable_wallets)
        target_address: Address to track (wallet or token)
    """
    # Get user from database
    user = get_user(update.effective_user.id)
    
    # Validate address format
    if not await is_valid_address(target_address):
        address_type = "token" if tracking_type == "token_profitable_wallets" else "wallet"
        await update.message.reply_text(
            f"❌ <b>Invalid {address_type.capitalize()} Address</b>\n\n"
            f"Please provide a valid Ethereum {address_type} address starting with 0x and containing 42 characters.",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Handle token profitable wallets tracking differently
    if tracking_type == "token_profitable_wallets":
        await handle_profitable_wallets_tracking(update, context, user, target_address)
        return
    
    # Create tracking subscription
    subscription = TrackingSubscription(
        user_id=user.user_id,
        tracking_type=tracking_type,
        target_address=target_address,
        is_active=True,
        created_at=datetime.now()
    )
    
    # Save subscription
    save_tracking_subscription(subscription)
    
    # Prepare confirmation message
    tracking_type_display = {
        "wallet_trades": "buy/sell activity",
        "token_deployments": "token deployments"
    }
    
    display_type = tracking_type_display.get(tracking_type, tracking_type)
    
    # Confirm to user
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="tracking_and_monitoring")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"✅ <b>Tracking Set Up Successfully!</b>\n\n"
        f"Now tracking {display_type} for wallet:\n"
        f"`{target_address[:6]}...{target_address[-4:]}`\n\n"
        f"You will receive notifications when this wallet makes significant trades or deploys new tokens.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )

async def handle_profitable_wallets_tracking(update: Update, context: ContextTypes.DEFAULT_TYPE, user: User, token_address: str) -> None:
    """
    Handle tracking profitable wallets for a token
    
    Args:
        update: The update object
        context: The context object
        user: User object
        token_address: Token address to track
    """
    # Send processing message
    processing_message = await update.message.reply_text(
        "🔍 Finding profitable wallets for this token... This may take a moment."
    )
    
    try:
        # Get token info
        token_info = await get_token_info(token_address)
        
        if not token_info:
            await processing_message.edit_text(
                "❌ Could not find information for this token. Please check the address and try again."
            )
            return
        
        # Get profitable wallets for this token
        profitable_wallets = await get_token_profitable_wallets(token_address, limit=5)
        
        if not profitable_wallets:
            await processing_message.edit_text(
                "❌ Could not find profitable wallets for this token at this time."
            )
            return
        
        # Create tracking subscription for the token
        token_subscription = TrackingSubscription(
            user_id=user.user_id,
            tracking_type="token_profitable_wallets",
            target_address=token_address,
            is_active=True,
            created_at=datetime.now()
        )
        
        # Save token subscription
        save_tracking_subscription(token_subscription)
        
        # Also track the top profitable wallets individually
        for wallet in profitable_wallets:
            wallet_subscription = TrackingSubscription(
                user_id=user.user_id,
                tracking_type="wallet_trades",
                target_address=wallet["address"],
                is_active=True,
                created_at=datetime.now()
            )
            save_tracking_subscription(wallet_subscription)
        
        # Format the response
        response = (
            f"✅ <b>Tracking Set Up Successfully!</b>\n\n"
            f"Now tracking profitable wallets for token:\n"
            f"<b>{token_info.get('name', 'Unknown')}</b> ({token_info.get('symbol', 'N/A')})\n"
            f"`{token_address[:6]}...{token_address[-4:]}`\n\n"
            f"<b>Top 5 Profitable Wallets Being Tracked:</b>\n"
        )
        
        for i, wallet in enumerate(profitable_wallets[:5], 1):
            response += (
                f"{i}. `{wallet['address'][:6]}...{wallet['address'][-4:]}`\n"
                f"   Profit: ${wallet.get('total_profit', 'N/A'):,.2f}\n"
                f"   Win Rate: {wallet.get('win_rate', 'N/A')}%\n\n"
            )
        
        response += "You will receive notifications when these wallets make significant trades with this token."
        
        # Add button to go back
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="tracking_and_monitoring")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await processing_message.edit_text(
            response,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    except Exception as e:
        logging.error(f"Error in handle_profitable_wallets_tracking: {e}")
        await processing_message.edit_text(
            "❌ An error occurred while setting up tracking. Please try again later."
        )

async def handle_view_tracking_subscriptions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle view tracking subscriptions button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Get user's tracking subscriptions
    subscriptions = get_user_tracking_subscriptions(user.user_id)
    
    if not subscriptions:
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="tracking_and_monitoring")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "📭 <b>No Active Tracking Subscriptions</b>\n\n"
            "You don't have any active tracking subscriptions at the moment.\n\n"
            "Use the Tracking & Monitoring menu to set up tracking for wallets, token deployments, or profitable wallets.",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    # Group subscriptions by type
    wallet_trades = [sub for sub in subscriptions if sub.tracking_type == "wallet_trades"]
    token_deployments = [sub for sub in subscriptions if sub.tracking_type == "token_deployments"]
    token_profitable_wallets = [sub for sub in subscriptions if sub.tracking_type == "token_profitable_wallets"]
    
    # Format the response
    response = (
        f"🔔 <b>Your Active Tracking Subscriptions</b>\n\n"
        f"Here’s a detailed overview of the wallets and activities you're currently monitoring using DeFi-Scope’s tracking system.\n"
        f"Below is a breakdown of your active subscriptions categorized by type:\n\n"
    )

    if wallet_trades:
        response += (
            f"📥 <b>Wallet Buy/Sell Activity Subscriptions ({len(wallet_trades)}):</b>\n"
            f"You're currently tracking buy and sell activities for the following wallets. Whenever any of these wallets perform a trade, you'll receive real-time alerts to stay informed and ahead of the curve:\n"
        )
        for i, sub in enumerate(wallet_trades[:3], 1):
            response += f"{i}. `{sub.target_address[:6]}...{sub.target_address[-4:]}`\n"
        if len(wallet_trades) > 3:
            response += f"   ...and <b>{len(wallet_trades) - 3}</b> more wallets are being tracked.\n"
        response += "\n"

    if token_deployments:
        response += (
            f"🧬 <b>Token Deployment Trackers ({len(token_deployments)}):</b>\n"
            f"These are the wallets you're monitoring for new token deployments. You'll be instantly notified when these addresses create new tokens, helping you catch early launches and analyze dev activity:\n"
        )
        for i, sub in enumerate(token_deployments[:3], 1):
            response += f"{i}. `{sub.target_address[:6]}...{sub.target_address[-4:]}`\n"
        if len(token_deployments) > 3:
            response += f"   ...and <b>{len(token_deployments) - 3}</b> more wallets are under watch.\n"
        response += "\n"

    if token_profitable_wallets:
        response += (
            f"📊 <b>Profitable Wallets Tracking ({len(token_profitable_wallets)}):</b>\n"
            f"You're currently tracking tokens to monitor which wallets are earning the most profits. This helps you identify smart money, trends in trading behavior, and potentially profitable plays:\n"
        )
        for i, sub in enumerate(token_profitable_wallets[:3], 1):
            token_info = await get_token_info(sub.target_address)
            token_name = token_info.get('symbol', 'Unknown') if token_info else 'Unknown'
            response += f"{i}. {token_name}: `{sub.target_address[:6]}...{sub.target_address[-4:]}`\n"
        if len(token_profitable_wallets) > 3:
            response += f"   ...and <b>{len(token_profitable_wallets) - 3}</b> more token contracts are actively being tracked.\n"
        response += "\n"

    
    # Add buttons to manage subscriptions
    keyboard = [
        [InlineKeyboardButton("🔍 Manage Wallet Traching for Buy/Sell Activity", callback_data="manage_wallet_tracking")],
        [InlineKeyboardButton("🧬 Manage Tracking for Token Deployment of a wallet", callback_data="manage_deployment_tracking")],
        [InlineKeyboardButton("📊 Manage Tracking for Profitable Wallets of a token", callback_data="manage_token_tracking")],
        [InlineKeyboardButton("🔙 Back", callback_data="tracking_and_monitoring")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        response,
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )

async def handle_manage_wallet_tracking(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle manage wallet tracking button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Get user's tracking subscriptions
    subscriptions = get_user_tracking_subscriptions(user.user_id)
    
    # Filter subscriptions by type
    wallet_subscriptions = [sub for sub in subscriptions if sub.tracking_type == "wallet_trades"]
    
    if not wallet_subscriptions:
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="view_tracking_subscriptions")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "📭 <b>No Wallet Tracking Subscriptions</b>\n\n"
            "You don't have any active wallet tracking subscriptions at the moment.",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    # Format the response
    response = f"📥 <b>Manage Wallet Tracking for buy/sell activities</b>\n\n"
    response += (
        f"You're currently tracking several wallets for buy and sell activities.\n"
        f"Below is the list of tracked wallets. If you wish to stop receiving alerts for any of them, simply select the wallet you'd like to remove:\n\n"
    )

    # Create keyboard with buttons to remove each subscription
    keyboard = []
    for i, sub in enumerate(wallet_subscriptions, 1):
        response += (
            f"{i}. `{sub.target_address[:6]}...{sub.target_address[-4:]}` — "
            f"Click the button below to stop tracking this wallet.\n"
        )
        keyboard.append([InlineKeyboardButton(
            f"❌ Remove #{i}", callback_data=f"remove_tracking_{sub.target_address}"
        )])
    
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="view_tracking_subscriptions")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        response,
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )

async def handle_manage_deployment_tracking(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle manage deployment tracking button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Get user's tracking subscriptions
    subscriptions = get_user_tracking_subscriptions(user.user_id)
    
    # Filter subscriptions by type
    deployment_subscriptions = [sub for sub in subscriptions if sub.tracking_type == "token_deployments"]
    
    if not deployment_subscriptions:
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="view_tracking_subscriptions")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "📭 <b>No Deployment Tracking Subscriptions</b>\n\n"
            "You don't have any active token deployment tracking subscriptions at the moment.",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    # Format the response
    response = f"🧬 <b>Manage Token Deployment Tracking</b>\n\n"
    response += f"Select a wallet to remove from Token deployment tracking:\n\n"
    
    # Create keyboard with buttons to remove each subscription
    keyboard = []
    for i, sub in enumerate(deployment_subscriptions, 1):
        response += f"{i}. `{sub.target_address[:6]}...{sub.target_address[-4:]}`\n"
        keyboard.append([InlineKeyboardButton(
            f"Remove #{i}", callback_data=f"remove_tracking_{sub.target_address}"
        )])
    
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="view_tracking_subscriptions")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        response,
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )

async def handle_manage_token_tracking(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle manage token tracking button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Get user's tracking subscriptions
    subscriptions = get_user_tracking_subscriptions(user.user_id)
    
    # Filter subscriptions by type
    token_subscriptions = [sub for sub in subscriptions if sub.tracking_type == "token_profitable_wallets"]
    
    if not token_subscriptions:
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="view_tracking_subscriptions")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "📭 <b>No Token Tracking Subscriptions</b>\n\n"
            "You don't have any active token profitable wallets tracking subscriptions at the moment.",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    # Format the response
    response = f"📊 <b>Manage Tracking for Profitable Wallets of a Token</b>\n\n"
    response += f"Select a token to remove from profitable wallets tracking:\n\n"
    
    # Create keyboard with buttons to remove each subscription
    keyboard = []
    for i, sub in enumerate(token_subscriptions, 1):
        # Get token info
        token_info = await get_token_info(sub.target_address)
        token_name = f"{token_info.get('name', 'Unknown')} ({token_info.get('symbol', 'N/A')})" if token_info else 'Unknown Token'
        
        response += f"{i}. {token_name}\n   `{sub.target_address[:6]}...{sub.target_address[-4:]}`\n"
        keyboard.append([InlineKeyboardButton(
            f"Remove #{i}", callback_data=f"remove_tracking_{sub.id}"
        )])
    
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="view_tracking_subscriptions")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        response,
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )

async def handle_remove_tracking(update: Update, context: ContextTypes.DEFAULT_TYPE, target_address: str) -> None:
    """
    Handle remove tracking button callback
    
    This function removes all tracking subscriptions for the given target address.
    """
    query = update.callback_query
    user = await check_callback_user(update)
        
    try:
        # Get all user's tracking subscriptions
        subscriptions = get_user_tracking_subscriptions(user.user_id)
        
        # Filter subscriptions for the target address
        matching_subs = [sub for sub in subscriptions if sub.target_address.lower() == target_address.lower()]
        
        if not matching_subs:
            await query.answer("No matching tracking subscription found.", show_alert=True)
            return
        
        # Delete each matching subscription
        for sub in matching_subs:
            delete_tracking_subscription(user.user_id, sub.tracking_type, sub.target_address)
        
        await query.answer("Tracking subscription(s) removed successfully!")
        
        # Redirect back to view tracking subscriptions
        await handle_view_tracking_subscriptions(update, context)
            
    except Exception as e:
        logging.error(f"Error removing tracking subscription: {e}")
        await query.answer("Failed to remove tracking subscription. Please try again.", show_alert=True)


# KOL wallets handlers
async def handle_kol_wallet_profitability(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle KOL wallet profitability button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    has_reached_limit, current_count = await check_rate_limit_service(
        user.user_id, "kol_wallet_profitability_scan", FREE_TOKEN_SCANS_DAILY
    )
    
    if has_reached_limit and not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back", callback_data="kol_wallets")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            f"⚠️ <b>Daily Limit Reached</b>\n\n"
            f"🧾 You've used <b>{current_count}</b> out of <b>{FREE_TOKEN_SCANS_DAILY}</b> free daily scans for <b>KOL Wallet Analysis</b>.\n"
            f"This feature lets you track the performance of known wallets and their trading behavior.\n\n"
            f"💎 <b>Upgrade to Premium</b> for unlimited scans and deeper DeFi intelligence.",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="kol_wallets")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.message.reply_text(
        "📢 <b>KOL Wallet Profitability Analysis</b>\n\n"
        "Please enter the name of the Key Opinion Leader(KOL) wallet you want to analyze.\n\n"
        "Example: `Binance`, `Alameda`, etc.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )
    
    context.user_data["expecting"] = "kol_wallet_name"

async def handle_track_whale_wallets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle track whale wallets button callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Check if user is premium
    if not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back", callback_data="kol_wallets")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "⭐ <b>Premium Feature</b>\n\n"
            "Tracking whale wallets is only available to premium users.\n\n"
            "Upgrade to premium to unlock all features!",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="kol_wallets")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Prompt user to enter token address
    await query.edit_message_text(
        "Please send me the token contract address to track its whale wallets.\n\n"
        "Example: `0x1234...abcd`\n\n"
        "I'll set up tracking for the top holders of this token.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )
    
    # Set conversation state to expect token address for whale tracking
    context.user_data["expecting"] = "track_whale_wallets_token"

async def handle_kol_period_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle selection of time period for KOL wallet profitability analysis"""
    query = update.callback_query
    
    kol_wallet_name = context.user_data.get("kol_wallet_name")
    
    if not kol_wallet_name:
        await query.answer("Please provide a Key Opinion Leader(KOL) wallet name first", show_alert=True)
        await handle_kol_wallet_profitability(update, context)
        return
    
    selected_period = int(query.data.replace("kol_period_", ""))
    
    async def get_kol_data_with_name(days, limit, chain):
        return await get_kol_wallet_profitability(
            days=days,
            limit=limit,
            chain=chain,
            kol_name=kol_wallet_name
        )
    
    await handle_period_selection_callback(
        update=update,
        context=context,
        get_data_func=get_kol_data_with_name,  
        format_response_func=format_kol_wallet_profitability_response,
        scan_count_type="kol_wallet_profitability_scan",
        processing_message_text=f"🔍 Analyzing {kol_wallet_name} wallet profitability over the last {selected_period} days... This may take a moment.",
        error_message_text=f"❌ An error occurred while analyzing {kol_wallet_name} wallet. Please try again later.",
        no_data_message_text=f"❌ Could not find profitability data for {kol_wallet_name} wallet in this period.",
        free_limit=FREE_RESPONSE_DAILY,
        premium_limit=PREMIUM_RESPONSE_DAILY
    )
    
    # Clear the KOL wallet name after processing
    if "kol_wallet_name" in context.user_data:
        del context.user_data["kol_wallet_name"]



async def handle_premium_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle premium info callback"""
    query = update.callback_query
    
    user = await check_callback_user(update)
    
    if user.is_premium:
        premium_until = user.premium_until.strftime("%d %B %Y") if user.premium_until else "Unknown"
        
        await query.edit_message_text(
            f"✨ <b>You're Already a Premium User!</b>\n\n"
            f"Thank you for supporting Crypto DeFi Analyze Bot.\n\n"
            f"Your premium subscription is active until: <b>{premium_until}</b>\n\n"
            f"Enjoy all the premium features!",
            parse_mode=ParseMode.HTML
        )
        return
    
    premium_text = (
        "⭐ <b>Upgrade to Crypto DeFi Analyze Premium</b>\n\n"

        "<b>🚀 Why Go Premium?</b>\n"
        "Gain unlimited access to powerful tools that help you track tokens, analyze wallets, "
        "and monitor whales like a pro. With Crypto DeFi Analyze Premium, you'll stay ahead of the market and "
        "make smarter investment decisions.\n\n"

        "<b>🔥 Premium Benefits:</b>\n"
        "✅ <b>Unlimited Token & Wallet Scans:</b> Analyze as many tokens and wallets as you want, with no daily limits.\n"
        "✅ <b>Deployer Wallet Analysis:</b> Find the deployer of any token, check their past projects, "
        "and spot potential scams before investing.\n"
        "✅ <b>Track Token, Wallet & Deployer Movements:</b> Get real-time alerts when a wallet buys, sells, "
        "or deploys a new token.\n"
        "✅ <b>View Top Holders of Any Token:</b> Discover which whales and big investors are holding a token, "
        "and track their transactions.\n"
        "✅ <b>Profitable Wallets Database:</b> Get exclusive access to a database of wallets that consistently "
        "make profits in the DeFi market.\n"
        "✅ <b>High Net Worth Wallet Monitoring:</b> Find wallets with high-value holdings and see how they invest.\n"
        "✅ <b>Priority Support:</b> Get faster responses and priority assistance from our support team.\n\n"

        "<b>💰 Premium Pricing Plans:</b>\n"
        "📅 <b>Weekly Plan:</b>\n"
        "• 0.1 ETH per week\n"
        "• 0.35 BNB per week\n\n"
        "📅 <b>Monthly Plan:</b>\n"
        "• 0.25 ETH per month\n"
        "• 1.0 BNB per month\n\n"

        "🔹 <b>Upgrade now</b> to unlock the full power of Crypto DeFi Analyze and take control of your investments!\n"
        "Select a plan below to get started:"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("🟢 Weekly - 🦄 0.1 ETH", callback_data="premium_plan_weekly_eth"),
            InlineKeyboardButton("🟢 Weekly - 🟡 0.35 BNB", callback_data="premium_plan_weekly_bnb")
        ],
        [
            InlineKeyboardButton("📅 Monthly - 🦄 0.25 ETH", callback_data="premium_plan_monthly_eth"),
            InlineKeyboardButton("📅 Monthly - 🟡 1.0 BNB", callback_data="premium_plan_monthly_bnb")
        ],
        [
            InlineKeyboardButton("🔙 Back", callback_data="back")
        ],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    try:
        await query.edit_message_text(
            premium_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logging.error(f"Error in handle_premium_info: {e}")
        await query.message.reply_text(
            premium_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        try:
            await query.message.delete()
        except:
            pass

async def handle_premium_purchase(update: Update, context: ContextTypes.DEFAULT_TYPE, plan: str, currency: str) -> None:
    """Handle premium purchase callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Get all payment details from a single function call
    payment_details = get_plan_payment_details(plan, currency)
    
    # Extract needed values from payment details
    wallet_address = payment_details["wallet_address"]
    crypto_amount = payment_details["amount"]
    duration_days = payment_details["duration_days"]
    display_name = payment_details["display_name"]
    display_price = payment_details["display_price"]
    network = payment_details["network"]
    currency_code = payment_details["currency"]
    
    # Determine network name for display
    network_name = "Ethereum" if network.lower() == "eth" else "Binance Smart Chain"
    
    # Show payment instructions with QR code
    payment_text = (
        f"🛒 <b>{display_name} Premium Plan</b>\n\n"
        f"Price: {display_price}\n"
        f"Duration: {duration_days} days\n\n"
        f"<b>Payment Instructions:</b>\n\n"
        f"1. Send <b>exactly {crypto_amount} {currency_code}</b> to our wallet address:\n"
        f"`{wallet_address}`\n\n"
        f"2. After sending, click 'I've Made Payment' and provide your transaction ID/hash.\n\n"
        f"<b>Important:</b>\n"
        f"• Send only {currency_code} on the {network_name} network\n"
        f"• Other tokens or networks will not be detected\n"
        f"• Transaction must be confirmed on the blockchain to activate premium"
    )
    
    # Store plan information in user_data for later use
    context.user_data["premium_plan"] = plan
    context.user_data["payment_currency"] = currency
    context.user_data["crypto_amount"] = crypto_amount
    
    keyboard = [
        [InlineKeyboardButton("I've Made Payment", callback_data=f"payment_made_{plan}_{currency}")],
        [InlineKeyboardButton("🔙 Back", callback_data="premium_info")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        await query.edit_message_text(
            payment_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        
        # Optionally, send a QR code as a separate message for easier scanning
        try:
            import qrcode
            from io import BytesIO
            
            # Create QR code with the wallet address and amount
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            
            # Format QR data based on currency
            if network.lower() == "eth":
                qr_data = f"ethereum:{wallet_address}?value={crypto_amount}"
            else:
                qr_data = f"binance:{wallet_address}?value={crypto_amount}"
                
            qr.add_data(qr_data)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Save QR code to BytesIO
            bio = BytesIO()
            img.save(bio, 'PNG')
            bio.seek(0)
            
            # Send QR code as photo
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=bio,
                caption=f"Scan this QR code to pay {crypto_amount} {currency_code} to our wallet"
            )
        except ImportError:
            # QR code library not available, skip sending QR code
            pass
        
    except Exception as e:
        logging.error(f"Error in handle_premium_purchase: {e}")

        await query.message.reply_text(
            payment_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        
        try:
            await query.message.delete()
        except:
            pass

async def handle_payment_made(update: Update, context: ContextTypes.DEFAULT_TYPE, plan: str, currency: str) -> None:
    """
    Handle payment made callback for crypto payments
    
    This function verifies a crypto payment and updates the user's premium status
    if the payment is confirmed on the blockchain.
    """
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Show processing message
    await query.edit_message_text(
        "🔄 Verifying payment on the blockchain... This may take a moment."
    )
    
    try:
        # 1. Get transaction ID from user data
        transaction_id = context.user_data.get("transaction_id")
        
        # If no transaction ID is stored, prompt user to provide it
        if not transaction_id:
            # Create a conversation to collect transaction ID
            context.user_data["awaiting_transaction_id"] = True
            context.user_data["premium_plan"] = plan
            context.user_data["payment_currency"] = currency
            
            keyboard = [[InlineKeyboardButton("🔙 Cancel", callback_data="premium_info")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "📝 <b>Transaction ID Required</b>\n\n"
                f"Please send the transaction hash/ID of your {currency.upper()} payment.\n\n"
                "You can find this in your wallet's transaction history after sending the payment.",
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
            return
        
        # 2. Get payment details based on the plan and currency
        payment_details = get_plan_payment_details(plan, currency)
        
        expected_amount = payment_details["amount"]
        wallet_address = payment_details["wallet_address"]
        duration_days = payment_details["duration_days"]
        network = payment_details["network"]
        
        # 3. Verify the payment on the blockchain
        from services.payment import verify_crypto_payment
        
        verification_result = await verify_crypto_payment(
            transaction_id=transaction_id,
            expected_amount=expected_amount,
            wallet_address=wallet_address,
            network=network
        )
        
        # 4. Process verification result
        if verification_result["verified"]:
            # Calculate premium expiration date
            from datetime import datetime, timedelta
            now = datetime.now()
            premium_until = now + timedelta(days=duration_days)
            
            # Update user's premium status in the database
            from data.database import update_user_premium_status
            
            # Update user status
            update_user_premium_status(
                user_id=user.user_id,
                is_premium=True,
                premium_until=premium_until,
                plan=plan,
                payment_currency=currency,
                transaction_id=transaction_id
            )
            
            # Clear transaction data from user_data
            if "transaction_id" in context.user_data:
                del context.user_data["transaction_id"]
            
            # Log successful premium activation
            logging.info(f"Premium activated for user {user.user_id}, plan: {plan}, currency: {currency}, until: {premium_until}")
            
            # Create confirmation message with back button
            keyboard = [[InlineKeyboardButton("🔙 Back to Menu", callback_data="back")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Send confirmation to user
            await query.edit_message_text(
                f"✅ <b>Payment Verified - Premium Activated!</b>\n\n"
                f"Thank you for upgrading to Crypto DeFi Analyze Premium.\n\n"
                f"<b>Transaction Details:</b>\n"
                f"• Plan: {plan.capitalize()}\n"
                f"• Amount: {expected_amount} {currency.upper()}\n"
                f"• Transaction: {transaction_id[:8]}...{transaction_id[-6:]}\n\n"
                f"Your premium subscription is now active until: "
                f"<b>{premium_until.strftime('%d %B %Y')}</b>\n\n"
                f"Enjoy all the premium features!",
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
            
            # Optional: Send a welcome message with premium tips
            await send_premium_welcome_message(update, context, user, plan, premium_until)
            
        else:
            # Payment verification failed
            error_message = verification_result.get("error", "Unknown error")
            
            # Create helpful error message based on the specific error
            if "not found" in error_message.lower():
                error_details = (
                    "• Transaction not found on the blockchain\n"
                    "• The transaction may still be pending\n"
                    "• Double-check that you entered the correct transaction ID"
                )
            elif "wrong recipient" in error_message.lower():
                error_details = (
                    "• Payment was sent to the wrong wallet address\n"
                    "• Please ensure you sent to the correct address: "
                    f"`{wallet_address[:10]}...{wallet_address[-8:]}`"
                )
            elif "amount mismatch" in error_message.lower():
                received = verification_result.get("received", 0)
                error_details = (
                    f"• Expected payment: {expected_amount} {currency.upper()}\n"
                    f"• Received payment: {received} {currency.upper()}\n"
                    "• Please ensure you sent the exact amount"
                )
            elif "pending confirmation" in error_message.lower():
                error_details = (
                    "• Transaction is still pending confirmation\n"
                    "• Please wait for the transaction to be confirmed\n"
                    "• Try again in a few minutes"
                )
            else:
                error_details = (
                    "• Payment verification failed\n"
                    "• The transaction may be invalid or incomplete\n"
                    "• Please try again or contact support"
                )
            
            # Create keyboard with options
            keyboard = [
                [InlineKeyboardButton("Try Again", callback_data=f"payment_retry_{plan}_{currency}")],
                [InlineKeyboardButton("Contact Support", url="https://t.me/SeniorCrypto01")],
                [InlineKeyboardButton("🔙 Back", callback_data="premium_info")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Send error message to user
            await query.edit_message_text(
                f"❌ <b>Payment Verification Failed</b>\n\n"
                f"We couldn't verify your payment:\n\n"
                f"{error_details}\n\n"
                f"Transaction ID: `{transaction_id[:10]}...{transaction_id[-8:]}`\n\n"
                f"Please try again or contact support for assistance.",
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
    
    except Exception as e:
        # Handle exceptions gracefully
        logging.error(f"Payment verification error: {e}")
        
        # Create keyboard with options
        keyboard = [
            [InlineKeyboardButton("Try Again", callback_data=f"premium_plan_{plan}_{currency}")],
            [InlineKeyboardButton("Contact Support", url="https://t.me/SeniorCrypto01")],
            [InlineKeyboardButton("🔙 Back", callback_data="premium_info")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Send error message to user
        await query.edit_message_text(
            "❌ <b>Error Processing Payment</b>\n\n"
            "An error occurred while verifying your payment.\n"
            "Please try again or contact support for assistance.",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )

async def handle_payment_retry(update: Update, context: ContextTypes.DEFAULT_TYPE, plan: str, currency: str) -> None:
    """Handle payment retry callback"""
    query = update.callback_query
    
    # Clear the stored transaction ID
    if "transaction_id" in context.user_data:
        del context.user_data["transaction_id"]
    
    # Set up to collect a new transaction ID
    context.user_data["awaiting_transaction_id"] = True
    context.user_data["premium_plan"] = plan
    context.user_data["payment_currency"] = currency
    
    keyboard = [[InlineKeyboardButton("🔙 Cancel", callback_data="premium_info")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "📝 <b>New Transaction ID Required</b>\n\n"
        f"Please send the new transaction hash/ID of your {currency.upper()} payment.\n\n"
        "You can find this in your wallet's transaction history after sending the payment.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )

async def handle_transaction_id_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle transaction ID input from user"""
    # Check if we're awaiting a transaction ID
    if not context.user_data.get("awaiting_transaction_id"):
        return
    
    # Get the transaction ID from the message
    transaction_id = update.message.text.strip()
    
    # Basic validation - transaction IDs are typically hex strings starting with 0x
    if not (transaction_id.startswith("0x") and len(transaction_id) >= 66):
        await update.message.reply_text(
            "⚠️ <b>Invalid Transaction ID</b>\n\n"
            "The transaction ID should start with '0x' and be at least 66 characters long.\n"
            "Please check your wallet and send the correct transaction hash.",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Store the transaction ID
    context.user_data["transaction_id"] = transaction_id
    
    # Get the plan and currency from user_data
    plan = context.user_data.get("premium_plan")
    currency = context.user_data.get("payment_currency")
    
    if not plan or not currency:
        await update.message.reply_text(
            "❌ <b>Error</b>\n\n"
            "Could not find your subscription plan details. Please start over.",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Clear the awaiting flag
    context.user_data["awaiting_transaction_id"] = False
    
    # Send confirmation and start verification
    confirmation_message = await update.message.reply_text(
        f"✅ Transaction ID received: `{transaction_id[:8]}...{transaction_id[-6:]}`\n\n"
        f"Now verifying your payment on the {currency.upper()} blockchain...",
        parse_mode=ParseMode.HTML
    )
    
    # Create verification button
    keyboard = [
        [InlineKeyboardButton("Verify Payment", callback_data=f"payment_made_{plan}_{currency}")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Update the message with a button to start verification
    await confirmation_message.edit_text(
        f"✅ Transaction ID received: `{transaction_id[:8]}...{transaction_id[-6:]}`\n\n"
        f"Click the button below to verify your payment on the {currency.upper()} blockchain.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )

# handle kol wallets 
async def handle_track_wallet(update: Update, context: ContextTypes.DEFAULT_TYPE, wallet_address: str) -> None:
    """Handle track wallet callback"""
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Check if user is premium
    if not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back", callback_data="back")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "⭐ <b>Premium Feature</b>\n\n"
            "Wallet tracking is only available to premium users.\n\n"
            "💎 Upgrade to premium to unlock all features!",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    subscription = TrackingSubscription(
        user_id=user.user_id,
        tracking_type="wallet",
        target_address=wallet_address,
        is_active=True,
        created_at=datetime.now()
    )
    
    # Save subscription
    save_tracking_subscription(subscription)
    
    # Confirm to user
    await query.edit_message_text(
        f"✅ Now tracking wallet: `{wallet_address[:6]}...{wallet_address[-4:]}`\n\n"
        f"You will receive notifications when this wallet makes significant trades, "
        f"deploys new tokens, or performs other notable actions.",
        parse_mode=ParseMode.MARKDOWN
    )

    
async def handle_wallet_holding_duration_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle wallet holding duration input"""
    from utils import handle_wallet_analysis_input
    from data.database import get_wallet_holding_duration
    
    await handle_wallet_analysis_input(
        update=update,
        context=context,
        analysis_type="wallet_holding_duration",
        get_data_func=get_wallet_holding_duration,
        format_response_func=format_wallet_holding_duration_response,
        scan_count_type="wallet_scan",
        processing_message_text="🔍 Analyzing wallet holding duration... This may take a moment.",
        error_message_text="❌ An error occurred while analyzing the wallet. Please try again later.",
        no_data_message_text="❌ Could not find holding duration data for this wallet."
    )

async def handle_tokens_deployed_wallet_address_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle tokens deployed by wallet input"""
    from utils import handle_wallet_analysis_input
    from data.database import get_tokens_deployed_by_wallet
    
    await handle_wallet_analysis_input(
        update=update,
        context=context,
        analysis_type="tokens_deployed_by_wallet",
        get_data_func=get_tokens_deployed_by_wallet,
        format_response_func=format_tokens_deployed_response,
        scan_count_type="wallet_scan",
        processing_message_text="🔍 Finding tokens deployed by this wallet... This may take a moment.",
        error_message_text="❌ An error occurred while analyzing the wallet. Please try again later.",
        no_data_message_text="❌ Could not find any tokens deployed by this wallet."
    )

async def handle_period_selection(
    update: Update, 
    context: ContextTypes.DEFAULT_TYPE,
    feature_info:str, 
    scan_type: str,
    callback_prefix: str
) -> None:
    """
    Generic handler for period selection
    
    Args:
        update: The update object
        context: The context object

        scan_type: Name of the feature for rate limiting
        title: Title to display in the message
        callback_prefix: Prefix for callback data
    """
    query = update.callback_query
    user = await check_callback_user(update)
    
    # Check if user has reached daily limit
    has_reached_limit, current_count = await check_rate_limit_service(
        user.user_id, scan_type, FREE_WALLET_SCANS_DAILY
    )
    
    if has_reached_limit and not user.is_premium:
        keyboard = [
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="premium_info")],
            [InlineKeyboardButton("🔙 Back", callback_data="wallet_analysis")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            f"⚠️ <b>Daily Limit Reached</b>\n\n"
            f"You’ve already used <b>{current_count}</b> out of your <b>{FREE_WALLET_SCANS_DAILY}</b> free daily wallet scans available for today. 🚫\n\n"
            f"To unlock unlimited access to powerful wallet analysis features, upgrade to <b>Premium</b> and explore the full potential of on-chain intelligence. 💎\n",
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        return
    
    # If user has not reached limit or is premium, show time period options
    keyboard = [
        [
            InlineKeyboardButton("1 Day", callback_data=f"{callback_prefix}_1"),
            InlineKeyboardButton("7 Days", callback_data=f"{callback_prefix}_7"),
            InlineKeyboardButton("30 Days", callback_data=f"{callback_prefix}_30")
        ],
        [InlineKeyboardButton("🔙 Back", callback_data="wallet_analysis")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Get the selected chain
    selected_chain = context.user_data.get("selected_chain", "eth")
    
    await query.message.reply_text(
        f"🔍 <b>Analyzing {feature_info} on {selected_chain}</b>\n\n"
        f"To proceed with a more in-depth analysis, please choose the time period you'd like to examine. "
        f"This will help us provide insights that are both accurate and relevant to your needs.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )

