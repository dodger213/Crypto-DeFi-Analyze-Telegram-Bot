import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Bot configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_USER_IDS = list(map(int, os.getenv("ADMIN_USER_IDS", "").split(",")))

# Blockchain configuration
ETH_PROVIDER_URI = os.getenv("ETH_PROVIDER_URI")
BASE_PROVIDER_URI = os.getenv("BASE_PROVIDER_URI")
BSC_PROVIDER_URI = os.getenv("BSC_PROVIDER_URI")

ETH_WSS_PROVIDER_URI = os.getenv("ETH_WSS_PROVIDER_URI")
BSC_WSS_PROVIDER_URI = os.getenv("BSC_WSS_PROVIDER_URI")
BASE_WSS_PROVIDER_URI = os.getenv("BASE_WSS_PROVIDER_URI")

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
BSCSCAN_API_KEY = os.getenv("BSCSCAN_API_KEY")
CHAINLINK_ETH_USD_PRICE_FEED_ADDRESS = os.getenv("CHAINLINK_ETH_USD_PRICE_FEED_ADDRESS")
SUBSCRIPTION_WALLET_ADDRESS=os.getenv("SUBSCRIPTION_WALLET_ADDRESS")

# Database configuration
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "crypto_defi_analyze_bot")

# Rate limits for free users
FREE_TOKEN_SCANS_DAILY=os.getenv("FREE_TOKEN_SCANS_DAILY")
FREE_WALLET_SCANS_DAILY=os.getenv("FREE_WALLET_SCANS_DAILY")

FREE_RESPONSE_DAILY = os.getenv("FREE_RESPONSE_DAILY")
PREMIUM_RESPONSE_DAILY = os.getenv("PREMIUM_RESPONSE_DAILY")                                                                     