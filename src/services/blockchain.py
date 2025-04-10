import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
import random
from web3 import Web3
from web3.exceptions import InvalidAddress, ContractLogicError

from config import ETH_PROVIDER_URI, BASE_PROVIDER_URI, BSC_PROVIDER_URI

from datetime import datetime, timedelta

from data.database import get_all_active_tracking_subscriptions

from services.notification import (
    send_tracking_notification,
    format_wallet_activity_notification,
    format_token_deployment_notification,
    format_profitable_wallet_notification
)

from utils import get_token_info

# Configure web3 connection
w3_eth = Web3(Web3.HTTPProvider(ETH_PROVIDER_URI))
w3_base = Web3(Web3.HTTPProvider(BASE_PROVIDER_URI))
w3_bsc = Web3(Web3.HTTPProvider(BSC_PROVIDER_URI))

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
    chain: str = "eth",
    token_address: Optional[str] = None,
    from_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Get recent transactions for a wallet, optionally filtered by token
    
    Args:
        wallet_address: The wallet address to get transactions for
        chain: The blockchain network (eth, base, bsc)
        token_address: Optional token address to filter transactions
        from_time: Optional datetime to get transactions after
        
    Returns:
        List of transaction dictionaries
    """
    logging.info(f"Getting recent transactions for wallet {wallet_address} on {chain}")
    
    try:
        # Initialize Web3 connection
        w3 = get_web3_provider(chain)
        
        # Normalize addresses
        wallet_address = w3.to_checksum_address(wallet_address)
        if token_address:
            token_address = w3.to_checksum_address(token_address)
        
        # Calculate from_block based on from_time if provided
        from_block = 'latest'
        if from_time:
            # Estimate block number based on timestamp
            # This is an approximation - for Ethereum, blocks are ~13 seconds apart
            # For BSC, blocks are ~3 seconds apart
            # For Base, blocks are ~2 seconds apart
            seconds_ago = (datetime.now() - from_time).total_seconds()
            if chain == "eth":
                blocks_ago = int(seconds_ago / 13)
            elif chain == "bsc":
                blocks_ago = int(seconds_ago / 3)
            elif chain == "base":
                blocks_ago = int(seconds_ago / 2)
            else:
                blocks_ago = int(seconds_ago / 13)  # Default to ETH
                
            # Get current block number
            current_block = w3.eth.block_number
            from_block = max(0, current_block - blocks_ago)
        
        transactions = []
        
        # Get normal transactions
        normal_txs = await get_normal_transactions(wallet_address, chain, from_block)
        transactions.extend(normal_txs)
        
        # Get token transfers if token_address is provided or to get all token transfers
        token_txs = await get_token_transfers(wallet_address, chain, token_address, from_block)
        transactions.extend(token_txs)
        
        # Sort transactions by timestamp (newest first)
        transactions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return transactions
        
    except Exception as e:
        logging.error(f"Error getting recent transactions: {e}", exc_info=True)
        return []

async def get_normal_transactions(wallet_address: str, chain: str, from_block: Union[int, str]) -> List[Dict[str, Any]]:
    """
    Get normal (ETH/BNB/BASE) transactions for a wallet
    
    Args:
        wallet_address: The wallet address
        chain: The blockchain network
        from_block: Starting block number
        
    Returns:
        List of transaction dictionaries
    """
    w3 = get_web3_provider(chain)
    
    # Create filter for transactions sent from the wallet
    from_filter = w3.eth.filter({
        'fromBlock': from_block,
        'toBlock': 'latest',
        'address': wallet_address
    })
    
    # Create filter for transactions sent to the wallet
    to_filter = w3.eth.filter({
        'fromBlock': from_block,
        'toBlock': 'latest',
        'address': wallet_address
    })
    
    # Get transactions
    from_entries = w3.eth.get_filter_logs(from_filter.filter_id)
    to_entries = w3.eth.get_filter_logs(to_filter.filter_id)
    
    # Process transactions
    transactions = []
    
    for entry in from_entries + to_entries:
        tx_hash = entry.get('transactionHash', '').hex()
        
        # Skip if we've already processed this transaction
        if any(tx.get('hash') == tx_hash for tx in transactions):
            continue
            
        # Get full transaction details
        tx = w3.eth.get_transaction(tx_hash)
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        block = w3.eth.get_block(tx.blockNumber)
        
        # Create transaction object
        transaction = {
            'hash': tx_hash,
            'from': tx['from'],
            'to': tx.get('to'),
            'value': w3.from_wei(tx.get('value', 0), 'ether'),
            'timestamp': datetime.fromtimestamp(block.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'is_token_transfer': False,
            'is_contract_creation': tx.get('to') is None
        }
        
        # If this is a contract creation, add the contract address
        if transaction['is_contract_creation'] and receipt.get('contractAddress'):
            transaction['contract_address'] = receipt.contractAddress
            transaction['contract_type'] = 'Unknown'  # Would need further analysis to determine type
        
        transactions.append(transaction)
    
    return transactions

async def get_token_transfers(wallet_address: str, chain: str, token_address: Optional[str] = None, from_block: Union[int, str] = 'latest') -> List[Dict[str, Any]]:
    """
    Get ERC20 token transfers for a wallet
    
    Args:
        wallet_address: The wallet address
        chain: The blockchain network
        token_address: Optional token address to filter transfers
        from_block: Starting block number
        
    Returns:
        List of transaction dictionaries
    """
    w3 = get_web3_provider(chain)
    
    # ERC20 Transfer event signature
    transfer_event_signature = w3.keccak(text="Transfer(address,address,uint256)").hex()
    
    # Create filter parameters
    filter_params = {
        'fromBlock': from_block,
        'toBlock': 'latest',
        'topics': [transfer_event_signature]
    }
    
    # Add token address filter if provided
    if token_address:
        filter_params['address'] = token_address
    
    # Create filter
    event_filter = w3.eth.filter(filter_params)
    
    # Get logs
    logs = w3.eth.get_filter_logs(event_filter.filter_id)
    
    # Process logs
    transactions = []
    
    for log in logs:
        # Check if this transfer involves our wallet
        from_address = '0x' + log['topics'][1].hex()[-40:]
        to_address = '0x' + log['topics'][2].hex()[-40:]
        
        if wallet_address.lower() not in [from_address.lower(), to_address.lower()]:
            continue
        
        # Get transaction details
        tx_hash = log.get('transactionHash', '').hex()
        tx = w3.eth.get_transaction(tx_hash)
        block = w3.eth.get_block(log.blockNumber)
        
        # Get token details
        token_contract_address = log.address
        token_contract = w3.eth.contract(address=token_contract_address, abi=ERC20_ABI)
        
        try:
            token_symbol = token_contract.functions.symbol().call()
            token_decimals = token_contract.functions.decimals().call()
        except Exception as e:
            logging.warning(f"Error getting token details: {e}")
            token_symbol = "UNKNOWN"
            token_decimals = 18
        
        # Parse token amount
        token_amount = int(log['data'], 16) / (10 ** token_decimals)
        
        # Create transaction object
        transaction = {
            'hash': tx_hash,
            'from': from_address,
            'to': to_address,
            'value': 0,  # ETH value is 0 for token transfers
            'timestamp': datetime.fromtimestamp(block.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'is_token_transfer': True,
            'token_address': token_contract_address,
            'token_symbol': token_symbol,
            'amount': token_amount,
            'is_buy': to_address.lower() == wallet_address.lower()
        }
        
        transactions.append(transaction)
    
    return transactions


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


# token_analysis
async def get_token_first_buyers(token_address: str, chain:str) -> List[Dict[str, Any]]:
    """
    Placeholder function for getting the first buyers data for a specific token
    
    Args:
        token_address: The token contract address
    
    Returns:
        List of dictionaries containing first buyer data
    """
    logging.info(f"Placeholder: get_token_first_buyers called for {token_address}")
        
    # Generate some dummy first buyers data
    first_buyers = []
    for i in range(10):
        # Generate a random wallet address
        wallet = "0x" + ''.join(random.choices('0123456789abcdef', k=40))
        
        first_buyers.append({
            "address": wallet,
            "buy_amount": round(random.uniform(1000, 10000), 2),
            "buy_value": round(random.uniform(0.5, 5), 2),
            "pnl": round(random.uniform(-50, 300), 2)
        })
    
    return first_buyers

async def get_token_profitable_wallets(token_address: str, chain:str) -> List[Dict[str, Any]]:
    """
    Placeholder function for getting the most profitable wallets for a specific token
    
    Args:
        token_address: The token contract address
    
    Returns:
        List of dictionaries containing profitable wallet data
    """
    logging.info(f"Placeholder: get_token_profitable_wallets called for {token_address}")
    
    # Generate some dummy profitable wallets data
    profitable_wallets = []
    for i in range(10):
        # Generate a random wallet address
        wallet = "0x" + ''.join(random.choices('0123456789abcdef', k=40))
        
        buy_amount = round(random.uniform(5000, 50000), 2)
        sell_amount = round(buy_amount * random.uniform(0.7, 0.95), 2)
        profit = round(random.uniform(1000, 10000), 2)
        
        profitable_wallets.append({
            "address": wallet,
            "buy_amount": buy_amount,
            "sell_amount": sell_amount,
            "profit": profit,
            "roi": round(random.uniform(50, 500), 2)
        })
    
    return profitable_wallets

async def get_ath_data(token_address: str, chain:str) -> Dict[str, Any]:
    """
    Placeholder function for getting the ATH data for a specific token
    
    Args:
        token_address: The token contract address
    
    Returns:
        Dictionary containing token ATH data
    """
    
    logging.info(f"Placeholder: get_ath_data called for {token_address}")
    
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
    
    # Create ATH data dictionary
    ath_data = {
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
        "percent_from_ath": round((current_price / ath_price) * 100, 2),
        "days_since_ath": random.randint(1, 30),
        "ath_volume": round(random.uniform(100000, 10000000), 2)
    }
    
    return ath_data

async def get_deployer_wallet_scan_data(token_address: str, chain:str) -> Dict[str, Any]:
    """
    Placeholder function for getting deployer wallet data for a specific token
    
    Args:
        token_address: The token contract address
    
    Returns:
        Dictionary containing deployer wallet data
    """
  
    logging.info(f"Placeholder: get_deployer_wallet_scan_data called for {token_address}")
    
    # Generate a random deployer wallet address
    deployer_address = "0x" + ''.join(random.choices('0123456789abcdef', k=40))
    
    # Generate random token data for demonstration purposes
    now = datetime.now()
    
    # Generate list of tokens deployed by this wallet
    deployed_tokens = []
    for i in range(random.randint(3, 10)):
        token_address = "0x" + ''.join(random.choices('0123456789abcdef', k=40))
        token_name = f"Token {i+1}"
        token_symbol = f"TKN{i+1}"
        
        # Generate random price and market cap
        current_price = round(random.uniform(0.00001, 100), 6)
        market_cap = round(current_price * random.uniform(1000000, 10000000000), 2)
        
        # Generate random ATH data
        ath_multiplier = random.uniform(1.5, 10)
        ath_price = round(current_price * ath_multiplier, 6)
        ath_market_cap = round(market_cap * ath_multiplier, 2)
        
        # Generate random dates
        deploy_date = (now - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d")
        
        # Calculate x-multiplier (ATH price / initial price)
        initial_price = round(current_price / random.uniform(1, ath_multiplier), 8)
        x_multiplier = round(ath_price / initial_price, 2)
        
        # Create token data
        token_data = {
            "address": token_address,
            "name": token_name,
            "symbol": token_symbol,
            "current_price": current_price,
            "current_market_cap": market_cap,
            "ath_price": ath_price,
            "ath_market_cap": ath_market_cap,
            "deploy_date": deploy_date,
            "initial_price": initial_price,
            "x_multiplier": f"{x_multiplier}x",
            "status": random.choice(["Active", "Abandoned", "Rugpull", "Successful"])
        }
        
        deployed_tokens.append(token_data)
    
    # Sort by deploy date (newest first)
    deployed_tokens.sort(key=lambda x: x["deploy_date"], reverse=True)
    
    # Create deployer wallet data
    deployer_data = {
        "deployer_address": deployer_address,
        "tokens_deployed": len(deployed_tokens),
        "first_deployment_date": deployed_tokens[-1]["deploy_date"],
        "last_deployment_date": deployed_tokens[0]["deploy_date"],
        "success_rate": round(random.uniform(10, 100), 2),
        "avg_roi": round(random.uniform(-50, 500), 2),
        "rugpull_count": random.randint(0, 3),
        "risk_level": random.choice(["Low", "Medium", "High", "Very High"]),
        "deployed_tokens": deployed_tokens
    }
    
    return deployer_data

async def get_token_top_holders(token_address: str, chain:str) -> List[Dict[str, Any]]:
    """
    Placeholder function for getting top holders data for a specific token
    
    Args:
        token_address: The token contract address
    
    Returns:
        List of dictionaries containing top holder data
    """
    logging.info(f"Placeholder: get_token_holders called for {token_address}")
    
    # Generate some dummy top holders data
    top_holders = []
    total_supply = random.uniform(1000000, 1000000000)
    
    # Generate top 10 holders
    for i in range(10):
        # Generate a random wallet address
        wallet = "0x" + ''.join(random.choices('0123456789abcdef', k=40))
        
        # Calculate percentage (decreasing as rank increases)
        percentage = round(random.uniform(30, 5) / (i + 1), 2)
        
        # Calculate token amount based on percentage
        token_amount = round((percentage / 100) * total_supply, 2)
        
        # Calculate USD value
        token_price = random.uniform(0.0001, 0.1)
        usd_value = round(token_amount * token_price, 2)
        
        # Determine if it's a DEX or CEX
        is_exchange = random.choice([True, False])
        exchange_type = random.choice(["Uniswap V3", "Uniswap V2", "SushiSwap", "PancakeSwap"]) if is_exchange else None
        
        # Determine wallet type
        wallet_type = "Exchange" if is_exchange else random.choice(["Whale", "Investor", "Team", "Unknown"])
        
        top_holders.append({
            "rank": i + 1,
            "address": wallet,
            "token_amount": token_amount,
            "percentage": percentage,
            "usd_value": usd_value,
            "wallet_type": wallet_type,
            "exchange_name": exchange_type,
            "holding_since": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d"),
            "last_transaction": (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
        })
    
    return top_holders

async def get_high_net_worth_holders(token_address: str, chain:str) -> List[Dict[str, Any]]:
    """
    Placeholder function for getting high net worth holders data for a specific token
    
    Args:
        token_address: The token contract address
    
    Returns:
        List of dictionaries containing high net worth holder data
    """
    logging.info(f"Placeholder: get_high_net_worth_holders called for {token_address}")
    
    # Generate some dummy high net worth holders data
    high_net_worth_holders = []
    
    # Generate 8-12 high net worth holders
    for i in range(random.randint(8, 12)):
        # Generate a random wallet address
        wallet = "0x" + ''.join(random.choices('0123456789abcdef', k=40))
        
        # Calculate token amount
        token_amount = round(random.uniform(100000, 10000000), 2)
        
        # Calculate USD value (minimum $10,000)
        token_price = random.uniform(0.001, 0.1)
        usd_value = max(10000, round(token_amount * token_price, 2))
        
        # Generate portfolio data
        portfolio_size = random.randint(3, 20)
        avg_holding_time = random.randint(30, 365)
        
        # Generate success metrics
        success_rate = round(random.uniform(50, 95), 2)
        avg_roi = round(random.uniform(20, 500), 2)
        
        high_net_worth_holders.append({
            "address": wallet,
            "token_amount": token_amount,
            "usd_value": usd_value,
            "portfolio_size": portfolio_size,  # Number of different tokens held
            "avg_holding_time": avg_holding_time,  # Average days holding tokens
            "success_rate": success_rate,  # Percentage of profitable trades
            "avg_roi": avg_roi,  # Average ROI percentage
            "first_seen": (datetime.now() - timedelta(days=random.randint(100, 1000))).strftime("%Y-%m-%d"),
            "last_transaction": (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
        })
    
    # Sort by USD value (highest first)
    high_net_worth_holders.sort(key=lambda x: x["usd_value"], reverse=True)
    
    return high_net_worth_holders


# wallet analysis
async def get_wallet_data(wallet_address: str, chain: str = "eth") -> dict:
    """
    Get data for a wallet address
    
    Args:
        wallet_address: The wallet address to analyze
        chain: The blockchain network (eth, base, bsc)
        
    Returns:
        Dictionary containing wallet data
    """
    logging.info(f"Placeholder: get_wallet_data called for {wallet_address} on {chain}")
    
    now = datetime.now()
    
    wallet_data = {
        "address": wallet_address,
        "first_transaction_date": (now - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
        "total_transactions": random.randint(10, 1000),
        "total_tokens_held": random.randint(5, 50),
        "estimated_value": round(random.uniform(1000, 1000000), 2),
        "chain": chain,  # Include the chain in the response
        "transaction_count": {
            "buys": random.randint(10, 500),
            "sells": random.randint(10, 300),
            "transfers": random.randint(5, 100),
            "swaps": random.randint(5, 100)
        },
        "profit_stats": {
            "total_profit_usd": round(random.uniform(-10000, 100000), 2),
            "win_rate": round(random.uniform(30, 90), 2),
            "avg_holding_time_days": round(random.uniform(1, 30), 2),
            "best_trade_profit": round(random.uniform(1000, 50000), 2),
            "worst_trade_loss": round(random.uniform(-20000, -100), 2)
        }
    }
    
    return wallet_data

async def get_wallet_most_profitable_in_period(days: int = 30, limit: int = 10, chain: str = "eth") -> List[Dict[str, Any]]:
    """
    Get the most profitable wallets in a specific period
    
    Args:
        days: Number of days to look back
        limit: Maximum number of wallets to return
        chain: The blockchain network (eth, base, bsc)
        
    Returns:
        List of dictionaries containing wallet data
    """
    logging.info(f"Getting most profitable wallets for {days} days, limit {limit}, chain {chain}")
    
    try:
        # This would be replaced with actual blockchain analysis
        # For now, we'll simulate the data
        wallets = []
        
        for i in range(1, limit + 5):
            # Generate random wallet data with decreasing profit
            profit = round(100000 / i, 2)
            win_rate = round(90 - (i * 1.5), 1)
            trades = random.randint(10, 50)
            
            wallets.append({
                "address": f"0x{i}wallet{random.randint(1000, 9999)}",
                "total_profit": profit,
                "win_rate": win_rate,
                "trades_count": trades,
                "period_days": days,
                "chain": chain
            })
        
        # Sort by profit
        wallets.sort(key=lambda x: x["total_profit"], reverse=True)
        
        logging.info(f"Returning {len(wallets[:limit])} wallets")
        return wallets[:limit]
    except Exception as e:
        logging.error(f"Error getting most profitable wallets: {e}", exc_info=True)
        return []

async def get_most_profitable_token_deployer_wallets(days: int = 30, limit: int = 10, chain: str = "eth") -> list:
    """
    Get most profitable token deployer wallets
    
    Args:
        days: Number of days to analyze
        limit: Maximum number of wallets to return
        chain: The blockchain network (eth, base, bsc)
        
    Returns:
        List of dictionaries containing profitable deployer wallet data
    """   
    logging.info(f"Placeholder: get_most_profitable_token_deployer_wallets called for {days} days on {chain}")
    
    # Generate dummy deployer wallets data
    deployer_wallets = []
    
    for i in range(limit):
        wallet_address = "0x" + ''.join(random.choices('0123456789abcdef', k=40))
        
        wallet = {
            "address": wallet_address,
            "tokens_deployed": random.randint(1, 20),
            "successful_tokens": random.randint(1, 10),
            "total_profit": round(random.uniform(5000, 500000), 2),
            "success_rate": round(random.uniform(20, 90), 2),
            "avg_roi": round(random.uniform(50, 1000), 2),
            "chain": chain,
            "period_days": days
        }
        
        deployer_wallets.append(wallet)
    
    # Sort by total profit (descending)
    deployer_wallets.sort(key=lambda x: x["total_profit"], reverse=True)
    
    return deployer_wallets

async def get_wallet_holding_duration(wallet_address: str, chain: str = "eth") -> dict:
    """
    Get holding duration data for a wallet
    
    Args:
        wallet_address: The wallet address to analyze
        chain: The blockchain network (eth, base, bsc)
        
    Returns:
        Dictionary containing holding duration data
    """
  
    logging.info(f"Placeholder: get_wallet_holding_duration called for {wallet_address} on {chain}")
    
    # Get basic wallet data
    wallet_data = await get_wallet_data(wallet_address, chain)
    
    # Generate holding duration data
    holding_data = {
        "wallet_address": wallet_address,
        "avg_holding_time_days": wallet_data["profit_stats"]["avg_holding_time_days"],
        "chain": chain,
        "tokens_analyzed": random.randint(10, 50),
        "holding_distribution": {
            "less_than_1_day": round(random.uniform(5, 30), 2),
            "1_to_7_days": round(random.uniform(20, 50), 2),
            "7_to_30_days": round(random.uniform(10, 40), 2),
            "more_than_30_days": round(random.uniform(5, 30), 2)
        },
        "token_examples": []
    }
    
    # Generate example tokens with holding durations
    token_names = ["Ethereum", "Uniswap", "Chainlink", "Aave", "Compound", "Synthetix", "Pepe", "Shiba Inu"]
    token_symbols = ["ETH", "UNI", "LINK", "AAVE", "COMP", "SNX", "PEPE", "SHIB"]
    
    for i in range(5):
        idx = random.randint(0, len(token_names) - 1)
        holding_days = round(random.uniform(0.5, 60), 1)
        
        token_example = {
            "name": token_names[idx],
            "symbol": token_symbols[idx],
            "address": "0x" + ''.join(random.choices('0123456789abcdef', k=40)),
            "holding_days": holding_days,
            "profit": round(random.uniform(-5000, 10000), 2)
        }
        
        holding_data["token_examples"].append(token_example)
    
    return holding_data

async def get_tokens_deployed_by_wallet(wallet_address: str, chain: str = "eth") -> list:
    """
    Get tokens deployed by a wallet
    
    Args:
        wallet_address: The wallet address to analyze
        chain: The blockchain network (eth, base, bsc)
        
    Returns:
        List of dictionaries containing token data
    """   
    logging.info(f"Placeholder: get_tokens_deployed_by_wallet called for {wallet_address} on {chain}")
    
    # Generate dummy tokens data
    tokens = []
    now = datetime.now()
    
    token_names = ["Super", "Mega", "Ultra", "Hyper", "Rocket", "Moon", "Star", "Galaxy"]
    token_suffixes = ["Token", "Coin", "Finance", "Cash", "Swap", "Yield", "Dao", "AI"]
    
    for i in range(random.randint(3, 10)):
        # Generate random token name and symbol
        name_prefix = random.choice(token_names)
        name_suffix = random.choice(token_suffixes)
        token_name = f"{name_prefix} {name_suffix}"
        token_symbol = f"{name_prefix[:1]}{name_suffix[:1]}".upper()
        
        # Generate random dates and prices
        deploy_date = (now - timedelta(days=random.randint(10, 180))).strftime("%Y-%m-%d")
        current_price = round(random.uniform(0.00001, 10), 6)
        
        # Generate random market caps
        current_market_cap = round(current_price * random.uniform(100000, 10000000), 2)
        ath_multiplier = random.uniform(1.5, 20)
        ath_market_cap = round(current_market_cap * ath_multiplier, 2)
        
        token = {
            "address": "0x" + ''.join(random.choices('0123456789abcdef', k=40)),
            "name": token_name,
            "symbol": token_symbol,
            "deploy_date": deploy_date,
            "current_price": current_price,
            "current_market_cap": current_market_cap,
            "ath_market_cap": ath_market_cap,
            "ath_multiplier": round(ath_multiplier, 2),
            "chain": chain,
            "deployer": wallet_address
        }
        
        tokens.append(token)
    
    # Sort by deploy date (newest first)
    tokens.sort(key=lambda x: x["deploy_date"], reverse=True)
    
    return tokens


# kol wallet profitability
async def get_kol_wallet_profitability(days: int, limit: int, chain: str = "eth", kol_name: str = None) -> list:
    """
    Get KOL wallet profitability data (DUMMY IMPLEMENTATION)
    
    Args:
        days: Number of days to analyze
        limit: Maximum number of results to return
        chain: Blockchain to analyze
        kol_name: Name of the specific KOL to filter by (optional)
        
    Returns:
        List of KOL wallet profitability data
    """

    # List of mock KOL names
    kol_names = [
        "Vitalik Buterin", "CZ Binance", "SBF", "Arthur Hayes", 
        "Justin Sun", "Elon Musk", "Crypto Cobain", "DeFi Dad",
        "Crypto Messiah", "Crypto Whale", "DegenSpartan", "Tetranode",
        "Hsaka", "Cobie", "DCinvestor", "ChainLinkGod"
    ]
    
    # Generate random KOL wallet data
    kol_wallets = []
    
    # If kol_name is provided, filter the list to only include that name (case-insensitive)
    if kol_name:
        filtered_names = [name for name in kol_names if kol_name.lower() in name.lower()]
        # If no match found, add the provided name to ensure we return something
        if not filtered_names and kol_name.strip():
            filtered_names = [kol_name]
        kol_names = filtered_names
    
    for i in range(min(len(kol_names), limit + 5)):  # Generate a few extra to sort later
        # Create base wallet data
        total_profit = random.uniform(10000, 1000000)
        win_rate = random.uniform(40, 95)
        
        # Generate random address
        address = "0x" + "".join(random.choice("0123456789abcdef") for _ in range(40))
        
        # Calculate period profit based on days
        if days == 1:
            period_profit = total_profit * random.uniform(0.01, 0.1)  # 1-10% of total profit
        elif days == 7:
            period_profit = total_profit * random.uniform(0.1, 0.4)   # 10-40% of total profit
        else:  # 30 days
            period_profit = total_profit * random.uniform(0.4, 1.0)   # 40-100% of total profit
        
        # Generate recent trades
        recent_trades = []
        for j in range(random.randint(3, 8)):
            trade_date = datetime.now() - timedelta(days=random.randint(0, days))
            token_names = ["ETH", "BTC", "LINK", "UNI", "AAVE", "MKR", "SNX", "YFI", "COMP", "SUSHI"]
            action = "Buy" if random.random() > 0.4 else "Sell"
            
            recent_trades.append({
                "token": random.choice(token_names),
                "action": action,
                "amount": round(random.uniform(0.1, 100), 2),
                "value": round(random.uniform(1000, 50000), 2),
                "date": trade_date.strftime("%Y-%m-%d %H:%M")
            })
        
        # Create wallet object
        wallet = {
            "name": kol_names[i] if i < len(kol_names) else f"Unknown KOL {i}",
            "address": address,
            "total_profit": total_profit,
            "period_profit": period_profit,
            "win_rate": round(win_rate, 1),
            "total_trades": random.randint(50, 500),
            "avg_position_size": round(random.uniform(5000, 100000), 2),
            "chain": chain,
            "period": days,
            "recent_trades": recent_trades
        }
        
        kol_wallets.append(wallet)
    
    # Sort by period profit
    kol_wallets.sort(key=lambda x: x.get("period_profit", 0), reverse=True)
    
    # Limit results
    return kol_wallets[:limit]
