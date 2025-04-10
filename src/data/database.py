import logging
import random
from typing import Optional, Dict, List, Any, Union
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.collection import Collection

from config import MONGODB_URI, DB_NAME, SUBSCRIPTION_WALLET_ADDRESS
from data.models import User, UserScan, TokenData, WalletData, TrackingSubscription, KOLWallet
from services.payment import get_plan_payment_details

_db: Optional[Database] = None

def init_database() -> bool:
    """Initialize the database connection and set up indexes"""
    global _db
    
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI) 
        _db = client[DB_NAME]
        
        # Set up indexes for collections
        # Users collection
        _db.users.create_index([("user_id", ASCENDING)], unique=True)
        
        # User scans collection
        _db.user_scans.create_index([
            ("user_id", ASCENDING),
            ("scan_type", ASCENDING),
            ("date", ASCENDING)
        ], unique=True)
        
        # Token data collection
        _db.token_data.create_index([("address", ASCENDING)], unique=True)
        _db.token_data.create_index([("deployer", ASCENDING)])
        
        # Wallet data collection
        _db.wallet_data.create_index([("address", ASCENDING)], unique=True)
        _db.wallet_data.create_index([("is_kol", ASCENDING)])
        _db.wallet_data.create_index([("is_deployer", ASCENDING)])
        
        # Tracking subscriptions collection
        _db.tracking_subscriptions.create_index([
            ("user_id", ASCENDING),
            ("tracking_type", ASCENDING),
            ("target_address", ASCENDING)
        ], unique=True)
        
        # KOL wallets collection
        _db.kol_wallets.create_index([("address", ASCENDING)], unique=True)
        _db.kol_wallets.create_index([("name", ASCENDING)])
        
        server_info = client.server_info()
        logging.info(f"✅ Successfully connected to MongoDB version: {server_info.get('version')}")
        logging.info(f"✅ Using database: {DB_NAME}")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to initialize database: {e}")
        return False

def get_database() -> Database:
    """Get the database instance"""
    global _db
    if _db is None:
        init_database()
    return _db

def get_user(user_id: int) -> Optional[User]:
    """Get a user by ID"""
    db = get_database()
    user_data = db.users.find_one({"user_id": user_id})
    if user_data:
        return User.from_dict(user_data)
    return None

def save_user(user: User) -> None:
    """Save or update a user"""
    db = get_database()
    user_dict = user.to_dict()
    db.users.update_one(
        {"user_id": user.user_id},
        {"$set": user_dict},
        upsert=True
    )

def get_all_users() -> List[User]:
    """Get all users in the database"""
    db = get_database()
    users = db.users.find()
    return [User.from_dict(user) for user in users]

def get_user_counts() -> Dict[str, int]:
    """Get user count statistics"""
    db = get_database()
    now = datetime.now()
    
    # Calculate date thresholds
    today_start = datetime.combine(now.date(), datetime.min.time())
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)
    
    # Get counts
    total_users = db.users.count_documents({})
    premium_users = db.users.count_documents({"is_premium": True})
    active_today = db.users.count_documents({"last_active": {"$gte": today_start}})
    active_week = db.users.count_documents({"last_active": {"$gte": week_ago}})
    active_month = db.users.count_documents({"last_active": {"$gte": month_ago}})
    
    return {
        "total_users": total_users,
        "premium_users": premium_users,
        "active_today": active_today,
        "active_week": active_week,
        "active_month": active_month
    }


def update_user_activity(user_id: int) -> None:
    """Update user's last active timestamp"""
    db = get_database()
    db.users.update_one(
        {"user_id": user_id},
        {"$set": {"last_active": datetime.now()}}
    )

def set_premium_status(user_id: int, is_premium: bool, duration_days: int = 30) -> None:
    """Set a user's premium status"""
    db = get_database()
    premium_until = datetime.now() + timedelta(days=duration_days) if is_premium else None
    db.users.update_one(
        {"user_id": user_id},
        {"$set": {
            "is_premium": is_premium,
            "premium_until": premium_until
        }}
    )

def get_user_scan_count(user_id: int, scan_type: str, date: str) -> int:
    """Get the number of scans a user has performed of a specific type on a date"""
    db = get_database()
    scan_data = db.user_scans.find_one({
        "user_id": user_id,
        "scan_type": scan_type,
        "date": date
    })
    return scan_data.get("count", 0) if scan_data else 0

def increment_user_scan_count(user_id: int, scan_type: str, date: str) -> None:
    """Increment the scan count for a user"""
    db = get_database()
    db.user_scans.update_one(
        {
            "user_id": user_id,
            "scan_type": scan_type,
            "date": date
        },
        {"$inc": {"count": 1}},
        upsert=True
    )

def reset_user_scan_counts() -> None:
    """Reset all user scan counts (typically called daily)"""
    db = get_database()
    today = datetime.now().date().isoformat()
    # Delete all scan records except today's
    db.user_scans.delete_many({"date": {"$ne": today}})



def get_user_tracking_subscriptions(user_id: int) -> List[TrackingSubscription]:
    """Get all tracking subscriptions for a user"""
    db = get_database()
    subscriptions = db.tracking_subscriptions.find({
        "user_id": user_id,
        "is_active": True
    })
    return [TrackingSubscription.from_dict(sub) for sub in subscriptions]

def get_tracking_subscription(user_id: int, tracking_type: str, target_address: str) -> Optional[TrackingSubscription]:
    """Get a specific tracking subscription"""
    db = get_database()
    subscription = db.tracking_subscriptions.find_one({
        "user_id": user_id,
        "tracking_type": tracking_type,
        "target_address": target_address.lower()
    })
    if subscription:
        return TrackingSubscription.from_dict(subscription)
    return None

def save_tracking_subscription(subscription: TrackingSubscription) -> None:
    """Save or update a tracking subscription"""
    db = get_database()
    sub_dict = subscription.to_dict()
    sub_dict["target_address"] = sub_dict["target_address"].lower()  # Normalize address
    
    db.tracking_subscriptions.update_one(
        {
            "user_id": sub_dict["user_id"],
            "tracking_type": sub_dict["tracking_type"],
            "target_address": sub_dict["target_address"]
        },
        {"$set": sub_dict},
        upsert=True
    )

def delete_tracking_subscription(user_id: int, tracking_type: str, target_address: str) -> None:
    """Delete a tracking subscription"""
    db = get_database()
    db.tracking_subscriptions.delete_one({
        "user_id": user_id,
        "tracking_type": tracking_type,
        "target_address": target_address.lower()
    })

def cleanup_expired_premium() -> None:
    """Remove premium status from users whose premium has expired"""
    db = get_database()
    now = datetime.now()
    db.users.update_many(
        {
            "is_premium": True,
            "premium_until": {"$lt": now}
        },
        {"$set": {
            "is_premium": False,
            "premium_until": None
        }}
    )

def get_all_active_tracking_subscriptions() -> List[TrackingSubscription]:
    """Get all active tracking subscriptions across all users"""
    db = get_database()
    subscriptions = db.tracking_subscriptions.find({"is_active": True})
    return [TrackingSubscription.from_dict(sub) for sub in subscriptions]

def get_users_with_expiring_premium(days_left: List[int]) -> List[User]:
    """Get users whose premium subscription is expiring in the specified number of days"""
    db = get_database()
    now = datetime.now()
    
    # Calculate date ranges for the specified days left
    date_ranges = []
    for days in days_left:
        start_date = now + timedelta(days=days)
        end_date = start_date + timedelta(days=1)
        date_ranges.append({"premium_until": {"$gte": start_date, "$lt": end_date}})
    
    # Find users with premium expiring in any of the specified ranges
    users = db.users.find({
        "is_premium": True,
        "$or": date_ranges
    })
    
    return [User.from_dict(user) for user in users]

def set_user_admin_status(user_id: int, is_admin: bool) -> None:
    """Set a user's admin status"""
    db = get_database()
    db.users.update_one(
        {"user_id": user_id},
        {"$set": {"is_admin": is_admin}}
    )

def update_user_referral_code(user_id: int, referral_code: str) -> None:
    """Update a user's referral code"""
    db = get_database()
    db.users.update_one(
        {"user_id": user_id},
        {"$set": {"referral_code": referral_code}}
    )

def record_referral(referrer_id: int, referred_id: int) -> None:
    """Record a referral relationship"""
    db = get_database()
    
    # Create referral record
    db.referrals.update_one(
        {
            "referrer_id": referrer_id,
            "referred_id": referred_id
        },
        {
            "$set": {
                "referrer_id": referrer_id,
                "referred_id": referred_id,
                "date": datetime.now()
            }
        },
        upsert=True
    )
    
    # Update referrer's stats
    db.users.update_one(
        {"user_id": referrer_id},
        {"$inc": {"referral_count": 1}}
    )

def update_user_premium_status(
    user_id: int,
    is_premium: bool,
    premium_until: datetime,
    plan: str,
    payment_currency: str = "eth",
    transaction_id: str = None
) -> None:
    """
    Update a user's premium status in the database and record the transaction
    
    Args:
        user_id: The Telegram user ID
        is_premium: Whether the user has premium status
        premium_until: The date until which premium is active
        plan: The premium plan (weekly or monthly)
        payment_currency: The currency used for payment (eth or bnb)
        transaction_id: The payment transaction ID (optional)
    """
    try:
        # Get database connection
        db = get_database()
        
        # Update user premium status
        db.users.update_one(
            {"user_id": user_id},
            {"$set": {
                "is_premium": is_premium,
                "premium_until": premium_until,
                "premium_plan": plan,
                "payment_currency": payment_currency,
                "last_payment_id": transaction_id,
                "updated_at": datetime.now()
            }}
        )
        
        # Get payment details
        payment_details = get_plan_payment_details(plan, payment_currency)
        
        # Record the transaction
        db.transactions.insert_one({
            "user_id": user_id,
            "type": "premium_purchase",
            "plan_type": plan,
            "currency": payment_details["currency"],  # Already uppercase from get_plan_payment_details
            "amount": payment_details["amount"],
            "duration_days": payment_details["duration_days"],
            "network": payment_details["network"],
            "transaction_id": transaction_id,
            "date": datetime.now()
        })
        
        logging.info(f"Updated premium status for user {user_id}: premium={is_premium}, plan={plan}, currency={payment_currency}, until={premium_until}")
        
    except Exception as e:
        logging.error(f"Error updating user premium status: {e}")
        raise
