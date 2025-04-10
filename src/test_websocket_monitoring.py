import asyncio
import logging
import sys
from datetime import datetime
from web3 import Web3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Import the blockchain service functions
from services.blockchain import (
    monitor_address_transactions,
    # monitor_token_transfers,
    get_web3_ws_provider,
    ERC20_ABI
)

# Test addresses to monitor
# Replace these with addresses that are likely to have transactions
TEST_ETH_ADDRESS = "0xfc9928f6590d853752824b0b403a6ae36785e535"  # Vitalik's address
TEST_BSC_ADDRESS = "0x2e87ac5b1af0d5c418365df3f2cf91240964e514"  # Binance Hot Wallet
TEST_BASE_ADDRESS = "0x154631a624eb048a2a074d380cfbaa906c885e2e"  # Active Base address

# Test tokens to monitor
# Replace these with active tokens on each chain
TEST_ETH_TOKEN = "0x8bbf486a9f2b535c4ef09c665a807859cfe92d83"  # USDT on Ethereum
TEST_BSC_TOKEN = "0x49b4af8d1b90ba15bb168f30a86a648b1aec4444"  # USDT on BSC
TEST_BASE_TOKEN = "0x198d878c9fd1a561f8e60c1e7c3fe6601e326b13"  # USDC on Base

# Callback functions to process detected transactions
async def eth_transaction_callback(transaction):
    logging.info(f"ETH Transaction Detected: {transaction['hash']}")
    logging.info(f"From: {transaction['from']}")
    logging.info(f"To: {transaction['to']}")
    logging.info(f"Value: {transaction['value']} ETH")
    logging.info(f"Timestamp: {transaction['timestamp']}")
    logging.info(f"Status: {transaction.get('status', 'unknown')}")
    logging.info("-" * 50)

async def bsc_transaction_callback(transaction):
    logging.info(f"BSC Transaction Detected: {transaction['hash']}")
    logging.info(f"From: {transaction['from']}")
    logging.info(f"To: {transaction['to']}")
    logging.info(f"Value: {transaction['value']} BNB")
    logging.info(f"Timestamp: {transaction['timestamp']}")
    logging.info(f"Status: {transaction.get('status', 'unknown')}")
    logging.info("-" * 50)

async def base_transaction_callback(transaction):
    logging.info(f"BASE Transaction Detected: {transaction['hash']}")
    logging.info(f"From: {transaction['from']}")
    logging.info(f"To: {transaction['to']}")
    logging.info(f"Value: {transaction['value']} ETH")
    logging.info(f"Timestamp: {transaction['timestamp']}")
    logging.info(f"Status: {transaction.get('status', 'unknown')}")
    logging.info("-" * 50)

# async def token_transfer_callback(transaction):
#     logging.info(f"Token Transfer Detected: {transaction['hash']}")
#     logging.info(f"Token: {transaction['token_symbol']} ({transaction['token_address']})")
#     logging.info(f"From: {transaction['from']}")
#     logging.info(f"To: {transaction['to']}")
#     logging.info(f"Amount: {transaction['amount']} {transaction['token_symbol']}")
#     logging.info(f"Is Buy: {transaction['is_buy']}")
#     logging.info(f"Timestamp: {transaction['timestamp']}")
#     logging.info(f"Status: {transaction.get('status', 'unknown')}")
#     logging.info("-" * 50)

async def test_websocket_connection(chain):
    """Test if the WebSocket connection is working"""
    try:
        w3 = get_web3_ws_provider(chain)
        block_number = w3.eth.block_number
        logging.info(f"Successfully connected to {chain} network. Current block: {block_number}")
        return True
    except Exception as e:
        logging.error(f"Failed to connect to {chain} network: {e}")
        return False

async def run_tests():
    """Run all the WebSocket monitoring tests"""
    logging.info("Starting WebSocket monitoring tests...")
    
    # Test WebSocket connections
    eth_connected = await test_websocket_connection("eth")
    bsc_connected = await test_websocket_connection("bsc")
    base_connected = await test_websocket_connection("base")
    
    # Create tasks for monitoring
    tasks = []
    
    if eth_connected:
        # Monitor Ethereum address
        tasks.append(asyncio.create_task(
            monitor_address_transactions(TEST_ETH_ADDRESS, "eth", eth_transaction_callback)
        ))
        logging.info(f"Started monitoring ETH address: {TEST_ETH_ADDRESS}")
        
        # Monitor Ethereum token transfers
        # tasks.append(asyncio.create_task(
        #     monitor_token_transfers(TEST_ETH_ADDRESS, "eth", TEST_ETH_TOKEN, token_transfer_callback)
        # ))
        # logging.info(f"Started monitoring token transfers for {TEST_ETH_TOKEN} on ETH")
    
    # if bsc_connected:
    #     # Monitor BSC address
    #     tasks.append(asyncio.create_task(
    #         monitor_address_transactions(TEST_BSC_ADDRESS, "bsc", bsc_transaction_callback)
    #     ))
    #     logging.info(f"Started monitoring BSC address: {TEST_BSC_ADDRESS}")
        
    #     # Monitor BSC token transfers
    #     # tasks.append(asyncio.create_task(
    #     #     monitor_token_transfers(TEST_BSC_ADDRESS, "bsc", TEST_BSC_TOKEN, token_transfer_callback)
    #     # ))
    #     # logging.info(f"Started monitoring token transfers for {TEST_BSC_TOKEN} on BSC")
    
    # if base_connected:
    #     # Monitor Base address
    #     tasks.append(asyncio.create_task(
    #         monitor_address_transactions(TEST_BASE_ADDRESS, "base", base_transaction_callback)
    #     ))
    #     logging.info(f"Started monitoring BASE address: {TEST_BASE_ADDRESS}")
        
        # Monitor Base token transfers
        # tasks.append(asyncio.create_task(
        #     monitor_token_transfers(TEST_BASE_ADDRESS, "base", TEST_BASE_TOKEN, token_transfer_callback)
        # ))
        # logging.info(f"Started monitoring token transfers for {TEST_BASE_TOKEN} on BASE")
    
    logging.info("All monitoring tasks started. Waiting for transactions...")
    logging.info("Press Ctrl+C to stop the test.")
    
    try:
        # Run for a specific duration or until manually stopped
        await asyncio.sleep(600)  # Run for 10 minutes
    except asyncio.CancelledError:
        logging.info("Test cancelled.")
    finally:
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        
        # Wait for all tasks to be cancelled
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logging.info("All monitoring tasks stopped.")

# Alternative test function that only monitors one specific address/token
async def run_single_test(address, chain, token_address=None):
    """Run a test for a single address and optionally a token"""
    logging.info(f"Starting test for address {address} on {chain}")
    
    # Test WebSocket connection
    connected = await test_websocket_connection(chain)
    if not connected:
        logging.error(f"Cannot run test: Failed to connect to {chain} network")
        return
    
    tasks = []
    
    # Monitor address transactions
    tasks.append(asyncio.create_task(
        monitor_address_transactions(address, chain, eth_transaction_callback)
    ))
    logging.info(f"Started monitoring {chain} address: {address}")
    
    # Monitor token transfers if token_address is provided
    # if token_address:
    #     tasks.append(asyncio.create_task(
    #         monitor_token_transfers(address, chain, token_address, token_transfer_callback)
    #     ))
    #     logging.info(f"Started monitoring token transfers for {token_address} on {chain}")
    
    logging.info("Monitoring started. Waiting for transactions...")
    logging.info("Press Ctrl+C to stop the test.")
    
    try:
        # Run for a specific duration or until manually stopped
        await asyncio.sleep(600)  # Run for 10 minutes
    except asyncio.CancelledError:
        logging.info("Test cancelled.")
    finally:
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        
        # Wait for all tasks to be cancelled
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logging.info("All monitoring tasks stopped.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test WebSocket monitoring functions")
    parser.add_argument("--address", help="Specific address to monitor")
    parser.add_argument("--chain", choices=["eth", "bsc", "base"], default="eth", help="Blockchain to monitor")
    parser.add_argument("--token", help="Specific token address to monitor")
    
    args = parser.parse_args()
    
    if args.address:
        # Run test for specific address
        asyncio.run(run_single_test(args.address, args.chain, args.token))
    else:
        # Run all tests
        asyncio.run(run_tests())
