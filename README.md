# Crypto DeFi Analyze Telegram Bot

A powerful Telegram bot for analyzing DeFi tokens, tracking wallets, and discovering profitable opportunities in the decentralized finance space.

# 🆘 Welcome to **Crypto DeFi Analyze Bot! 🎉**

## 🔎 **Your Ultimate DeFi Intelligence Bot!**
Stay ahead in the crypto game with powerful analytics, wallet tracking, and market insights. 📊💰

<p align="center">
  <img src="https://github.com/dodger213/Crypto-DeFi-Analyze-Telegram-Bot/blob/main/src/image/main_menu.PNG" alt="Main Menu">
</p>

## ✨ **What can I do for you?**

### 📊 **Token Analysis**

- 🔹 **First Buyers & Profits of a Token:**  
  See the first 1-50 buy wallets of a token with buy & sell amount, trades, PNL, and win rate.  
  *(Free users: 3 token scans daily | Premium: Unlimited)*

- 🔹 **Most Profitable Wallets of a Token:**  
  Discover wallets with the highest profits in any token.  
  *(Free users: 3 token scans daily | Premium: Unlimited)*

- 🔹 **Market Cap & ATH:**  
  View all-time high (ATH) market cap, date, and % drop from ATH.  
  *(Free users: 3 token scans daily | Premium: Unlimited)*

- 🔹 **Deployer Wallet Scan:** *(Premium)*  
  Reveal the deployer wallet of any token and see other tokens deployed, their ATH market cap, and ROI.

- 🔹 **Top Holders & Whale Watch:** *(Premium)*  
  View top 10 holders and whale wallets of any token.

- 🔹 **High Net Worth Wallet Holders:** *(Premium)*  
  See wallets with $10,000+ holdings, total USD worth, token list, and holding durations.

---

### 🕵️ **Wallet Analysis**

- 🔹 **Most Profitable Wallets in a Time Period:**  
  Find wallets with highest profits in 1-30 days.  
  *(Free: 2 wallets per query | Premium: Unlimited)*

- 🔹 **Wallet Holding Duration:**  
  Analyze how long a wallet holds before selling.  
  *(Free: 3 scans daily | Premium: Unlimited)*

- 🔹 **Most Profitable Token Deployer Wallets:**  
  Spot high-profit deployers in 1-30 days.  
  *(Free: 2 deployers per query | Premium: Unlimited)*

- 🔹 **Tokens Deployed by Wallet:** *(Premium)*  
  See tokens deployed by any wallet, with name, ticker, current price, deployment date, market cap, and ATH.

---

### 🔔 **Tracking & Monitoring** *(Premium)*

- 🔹 **Track Buy/Sell Activity:**  
  Get alerts when a wallet buys or sells any token.

- 🔹 **Track New Token Deployments:**  
  Be notified when a wallet (or its linked wallets) deploy a new token.

- 🔹 **Profitable Wallets in a Token:**  
  Track most profitable wallets in any token within 1-30 days.

---

### 🐳 **KOL Wallets**

- 🔹 **KOL Wallets Profitability:**  
  Monitor top KOL wallets' performance and PNL in 1-30 days.  
  *(Free: 3 scans daily | Premium: Unlimited)*

- 🔹 **Track Whale Wallets:** *(Premium)*  
  Get notified when Devs, top holders, or whales sell a token.



### Premium Features

Upgrade to premium to unlock:

- Unlimited token and wallet scans
- Access to deployer wallet analysis
- Tracking tokens, wallets, and deployers
- View top holders of any token
- Access to profitable wallets database
- High net worth wallet monitoring
- Priority support

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/dodger213/Crypto-DeFi-Analyze-Telegram-Bot.git
cd Crypto-DeFi-Analyze-Telegram-Bot
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
MONGODB_URI="mongodb://localhost:27017/"

TELEGRAM_TOKEN="your telegram bot token"

FREE_TOKEN_SCANS_DAILY=3
FREE_WALLET_SCANS_DAILY=3

FREE_RESPONSE_DAILY = 2
PREMIUM_RESPONSE_DAILY = 10

WEB3_PROVIDER_URI="Your web3 provider uri"
ETHERSCAN_API_KEY="Your ethrescan api key"
```

4. Run the bot:
```
python bot/main.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the [MIT License](./LICENSE) - see the LICENSE file for details.

## Contact
Please contact me via [telegram](https://t.me/stevendev0822), if you have any problems.
