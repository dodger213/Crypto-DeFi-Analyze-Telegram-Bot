import logging
import asyncio
from typing import Dict, List, Optional, Any
import re
from web3 import Web3
from web3.exceptions import InvalidAddress, ContractLogicError

from config import WEB3_PROVIDER_URI

from datetime import datetime, timedelta

from data.database import (
    get_all_active_tracking_subscriptions,
    get_token_profitable_wallets,
)

from services.notification import (
    send_tracking_notification,
    format_wallet_activity_notification,
    format_token_deployment_notification,
    format_profitable_wallet_notification
)

# Configure web3 connection
w3_eth = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URI))
w3_base = Web3(Web3.HTTPProvider('https://mainnet.base.org'))
w3_bsc = Web3(Web3.HTTPProvider('https://bsc-dataseed.binance.org/'))

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
]

async def is_valid_address(address: str) -> bool:
    if not address:
        return False
    
    return Web3.is_address(address)

async def is_valid_token_contract(address: str, chain: str) -> bool:
    if not await is_valid_address(address):
        logging.warning(f"Invalid address format: {address}")
        return False

    w3 = get_web3_provider(chain)

    try:
        checksum_address = w3.to_checksum_address(address.lower())
        code = w3.eth.get_code(checksum_address)

        if code == b'' or code == b'0x':
            logging.info("Address has no contract code.")
            return False

        contract = w3.eth.contract(address=checksum_address, abi=ERC20_ABI)

        try:
            symbol = contract.functions.symbol().call()
            logging.info(f"Token symbol: {symbol}")
        except Exception as e:
            logging.warning(f"Couldn't get token symbol: {e}")

        try:
            name = contract.functions.name().call()
            logging.info(f"Token name: {name}")
            return True
        except Exception as e:
            logging.warning(f"Couldn't get token name: {e}")

        try:
            decimals = contract.functions.decimals().call()
            logging.info(f"Token decimals: {decimals}")
            return True
        except Exception as e:
            logging.warning(f"Couldn't get token decimals: {e}")

        logging.warning("Address has code but no ERC-20 behavior.")
        return False

    except Exception as e:
        logging.error(f"Error validating token contract: {e}")
        return False
    
async def is_valid_wallet_address(address: str, chain:str) -> bool:
    """
    Validate if the provided address is a wallet (not a contract)
    
    Args:
        address: The address to validate
        chain: The blockchain network (eth, base, bsc)
    
    Returns:
        bool: True if the address is a valid wallet, False otherwise
    """
    # First check if it's a valid address
    if not await is_valid_address(address):
        return False
    
    w3 = get_web3_provider(chain)
    
    try:
        checksum_address = w3.to_checksum_address(address.lower())
        code = w3.eth.get_code(checksum_address)
        # If there's no code, it's a regular wallet address
        return code == b'' or code == '0x'
    except Exception as e:
        logging.error(f"Error validating wallet address on {chain}: {e}")
        # Return True if the format is correct but web3 validation fails
        # This is a fallback to prevent false negatives due to connection issues
        return True

def get_web3_provider(chain: str):
    """
    Get the appropriate Web3 provider for the specified chain
    
    Args:
        chain: The blockchain network (eth, base, bsc)
    
    Returns:
        Web3: The Web3 provider for the specified chain
    """
    if chain == "eth":
        return w3_eth
    elif chain == "base":
        return w3_base
    elif chain == "bsc":
        return w3_bsc
    else:
        logging.warning(f"Unknown chain '{chain}', defaulting to Ethereum")
        return w3_eth

def check_providers():
    """Check if all blockchain providers are connected"""
    eth_connected = w3_eth.is_connected()
    base_connected = w3_base.is_connected()
    bsc_connected = w3_bsc.is_connected()
    
    if not (eth_connected and base_connected and bsc_connected):
        logging.warning(f"Provider connection status: ETH: {eth_connected}, BASE: {base_connected}, BSC: {bsc_connected}")
    
    return eth_connected, base_connected, bsc_connected


async def get_token_info(token_address: str, chain: str = "eth") -> Optional[Dict[str, Any]]:
    """Get detailed information about a token"""
    if not await is_valid_token_contract(token_address, chain):
        return None
    
    try:      
        # Get the appropriate web3 provider based on chain
        w3 = get_web3_provider(chain)
        
        # ERC20 ABI for basic token information
        abi = [
            {"constant": True, "inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "type": "function"}
        ]
        
        # Create contract instance
        checksum_address = w3.to_checksum_address(token_address)
        contract = w3.eth.contract(address=checksum_address, abi=abi)
        
        # Get basic token information
        name = contract.functions.name().call()
        symbol = contract.functions.symbol().call()
        decimals = contract.functions.decimals().call()
        total_supply = contract.functions.totalSupply().call() / (10 ** decimals)
        
        # Simulate historical data
        return {
            "address": token_address,
            "name": name,
            "symbol": symbol,
            "decimals": decimals,
            "total_supply": total_supply
        }
    except Exception as e:
        logging.error(f"Error getting token info on {chain}: {e}")
        return None

async def get_recent_transactions(
    wallet_address: str, 
    token_address: Optional[str] = None,
    from_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Get recent transactions for a wallet, optionally filtered by token
    
    Args:
        wallet_address: The wallet address to get transactions for
        token_address: Optional token address to filter transactions
        from_time: Optional datetime to get transactions after
        
    Returns:
        List of transaction dictionaries
    """
    logging.info(f"Getting recent transactions for wallet {wallet_address}")
    
    try:
        # Initialize Web3 connection
        w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URI))
        
        # Normalize addresses
        wallet_address = w3.to_checksum_address(wallet_address)
        if token_address:
            token_address = w3.to_checksum_address(token_address)
        
        # In a real implementation, you would:
        # 1. Query blockchain or indexer API for transactions
        # 2. Filter by token_address if provided
        # 3. Filter by timestamp if from_time is provided
        
        # For now, return mock data
        mock_transactions = []
        
        # Mock a token transfer transaction
        mock_token_tx = {
            'hash': f"0x{wallet_address[2:10]}000000000000000000000000",
            'from': wallet_address,
            'to': "0x1234567890123456789012345678901234567890",
            'value': 0,  # ETH value is 0 for token transfers
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_token_transfer': True,
            'token_address': token_address or "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC if none provided
            'token_symbol': "USDC",
            'amount': 1000.0,
            'value_usd': 1000.0,
            'is_buy': True
        }
        mock_transactions.append(mock_token_tx)
        
        # Mock a contract creation transaction
        mock_contract_tx = {
            'hash': f"0x{wallet_address[2:10]}111111111111111111111111",
            'from': wallet_address,
            'to': None,  # Contract creation has no 'to' address
            'value': 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_contract_creation': True,
            'contract_address': "0x9876543210987654321098765432109876543210",
            'contract_type': "ERC20"
        }
        mock_transactions.append(mock_contract_tx)
        
        # Filter by time if needed
        if from_time:
            # In a real implementation, you would filter by timestamp
            pass
        
        return mock_transactions
        
    except Exception as e:
        logging.error(f"Error getting recent transactions: {e}")
        return []

def is_token_transfer(tx: Dict[str, Any]) -> bool:
    """
    Check if a transaction is a token transfer
    
    Args:
        tx: Transaction dictionary
        
    Returns:
        True if the transaction is a token transfer, False otherwise
    """
    # In a real implementation, you would:
    # 1. Check if the transaction has token transfer event logs
    # 2. Look for ERC20 Transfer events
    
    # For now, use a simple flag in our mock data
    return tx.get('is_token_transfer', False)

def is_contract_creation(tx: Dict[str, Any]) -> bool:
    """
    Check if a transaction is a contract creation
    
    Args:
        tx: Transaction dictionary
        
    Returns:
        True if the transaction is a contract creation, False otherwise
    """
    # In a real implementation, you would:
    # 1. Check if 'to' is None or empty
    # 2. Verify that a contract address was created
    
    # For now, use a simple flag in our mock data
    return tx.get('is_contract_creation', False)

# Keep track of processed transactions to avoid duplicate notifications
processed_txs = set()

async def start_blockchain_monitor():
    """Start the blockchain monitor as a background task"""
    logging.info("Starting blockchain monitor...")
    asyncio.create_task(monitor_blockchain_events())

async def monitor_blockchain_events():
    """Background task to monitor blockchain events and send notifications"""
    logging.info("Blockchain monitor running")

    from utils import get_token_info
    
    # Keep track of processed transactions to avoid duplicate notifications
    processed_txs = set()
    
    while True:
        try:
            # Get all active tracking subscriptions
            subscriptions = get_all_active_tracking_subscriptions()
            
            if not subscriptions:
                # No active subscriptions, sleep and check again later
                await asyncio.sleep(60)
                continue
            
            # Group subscriptions by address for efficient querying
            tracked_wallets = {}
            tracked_tokens = {}
            
            for sub in subscriptions:
                if sub.tracking_type in ["wallet_trades", "token_deployments"]:
                    if sub.target_address not in tracked_wallets:
                        tracked_wallets[sub.target_address] = []
                    tracked_wallets[sub.target_address].append(sub)
                elif sub.tracking_type == "token_profitable_wallets":
                    if sub.target_address not in tracked_tokens:
                        tracked_tokens[sub.target_address] = []
                    tracked_tokens[sub.target_address].append(sub)
            
            # Check for new transactions for tracked wallets
            for wallet_address, subs in tracked_wallets.items():
                # Get transactions from the last 10 minutes
                recent_txs = await get_recent_transactions(
                    wallet_address, 
                    from_time=datetime.now() - timedelta(minutes=10)
                )
                
                for tx in recent_txs:
                    # Skip already processed transactions
                    tx_hash = tx.get('hash')
                    if tx_hash in processed_txs:
                        continue
                    
                    # Mark as processed
                    processed_txs.add(tx_hash)
                    
                    # Process each transaction
                    if is_token_transfer(tx):
                        # Notify users tracking wallet trades
                        for sub in subs:
                            if sub.tracking_type == "wallet_trades":
                                token_info = await get_token_info(tx['token_address'])
                                tx['token_name'] = token_info.get('symbol', 'Unknown Token')
                                
                                message = format_wallet_activity_notification(
                                    wallet_address=wallet_address,
                                    tx_data=tx
                                )
                                
                                await send_tracking_notification(sub.user_id, message)
                    
                    elif is_contract_creation(tx):
                        # Notify users tracking token deployments
                        for sub in subs:
                            if sub.tracking_type == "token_deployments":
                                message = format_token_deployment_notification(
                                    deployer_address=wallet_address,
                                    contract_address=tx['contract_address'],
                                    timestamp=tx['timestamp']
                                )
                                
                                await send_tracking_notification(sub.user_id, message)
            
            # Check for transactions involving tracked tokens
            for token_address, subs in tracked_tokens.items():
                # Get profitable wallets for this token
                profitable_wallets = await get_token_profitable_wallets(token_address)
                
                # Get token info once for all notifications
                token_info = await get_token_info(token_address)
                token_name = token_info.get('symbol', 'Unknown Token')
                
                # Check for recent transactions by these wallets
                for wallet in profitable_wallets:
                    wallet_address = wallet['address']
                    
                    # Get transactions involving this token from the last 10 minutes
                    recent_txs = await get_recent_transactions(
                        wallet_address=wallet_address,
                        token_address=token_address,
                        from_time=datetime.now() - timedelta(minutes=10)
                    )
                    
                    for tx in recent_txs:
                        # Skip already processed transactions
                        tx_hash = tx.get('hash')
                        if tx_hash in processed_txs:
                            continue
                        
                        # Mark as processed
                        processed_txs.add(tx_hash)
                        
                        # Notify users tracking profitable wallets
                        for sub in subs:
                            message = format_profitable_wallet_notification(
                                wallet_address=wallet_address,
                                token_name=token_name,
                                tx_data=tx
                            )
                            
                            await send_tracking_notification(sub.user_id, message)
            
            # Limit the size of processed_txs to prevent memory issues
            if len(processed_txs) > 10000:
                # Keep only the 5000 most recent transactions
                processed_txs_list = list(processed_txs)
                processed_txs = set(processed_txs_list[-5000:])
            
            # Sleep before next check
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logging.error(f"Error in blockchain monitor: {e}")
            # Sleep before retrying
            await asyncio.sleep(60)

