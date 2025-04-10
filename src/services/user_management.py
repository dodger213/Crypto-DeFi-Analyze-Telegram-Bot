import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from data.models import User
from data.database import (
    get_user, save_user, update_user_activity, get_user_scan_count,
    increment_user_scan_count, set_premium_status,
    cleanup_expired_premium, get_user_counts,
)

async def get_or_create_user(user_id: int, username: Optional[str] = None, 
                           first_name: Optional[str] = None, last_name: Optional[str] = None) -> User:
    """Get a user from the database or create if not exists"""
    user = get_user(user_id)
    
    if not user:
        # Create new user
        user = User(
            user_id=user_id,
            username=username,
            first_name=first_name,
            last_name=last_name
        )
        save_user(user)
    else:
        # Update user activity
        update_user_activity(user_id)
    
    return user

async def extend_premium_subscription(user_id: int, additional_days: int) -> bool:
    """Extend an existing premium subscription"""
    user = get_user(user_id)
    if not user:
        return False
    
    try:
        # Calculate new expiration date
        if user.is_premium and user.premium_until:
            # If already premium, add days to current expiration
            current_expiry = user.premium_until
            new_expiry = current_expiry + timedelta(days=additional_days)
            days_until_expiry = (new_expiry - datetime.now()).days
            set_premium_status(user_id, True, days_until_expiry)
        else:
            # If not premium, start new subscription
            set_premium_status(user_id, True, additional_days)
        
        return True
    except Exception as e:
        logging.error(f"Error extending premium subscription for user {user_id}: {e}")
        return False

async def check_rate_limit_service(user_id: int, scan_type: str, limit: int) -> Tuple[bool, int]:
    """
    Check if user has exceeded their daily scan limit
    Returns (has_reached_limit, current_count)
    """
    user = get_user(user_id)
    
    # Premium users have no limits
    if user and user.is_premium:
        return False, 0
    
    # Check scan count for today
    today = datetime.now().date().isoformat()
    scan_count = get_user_scan_count(user_id, scan_type, today)
    
    return scan_count >= limit, scan_count

async def increment_scan_count(user_id: int, scan_type: str) -> int:
    """Increment a user's scan count and return the new count"""
    today = datetime.now().date().isoformat()
    increment_user_scan_count(user_id, scan_type, today)
    return get_user_scan_count(user_id, scan_type, today)

async def get_user_premium_info(user_id: int) -> Dict[str, Any]:
    """Get information about a user's premium status"""
    user = get_user(user_id)
    if not user:
        return {
            "is_premium": False,
            "days_left": 0,
            "expiry_date": None
        }
    
    if not user.is_premium:
        return {
            "is_premium": False,
            "days_left": 0,
            "expiry_date": None
        }
    
    days_left = 0
    expiry_date = None
    
    if user.premium_until:
        days_left = max(0, (user.premium_until - datetime.now()).days)
        expiry_date = user.premium_until.strftime("%d %B %Y")
    
    return {
        "is_premium": user.is_premium,
        "days_left": days_left,
        "expiry_date": expiry_date
    }

async def get_user_usage_stats(user_id: int) -> Dict[str, Any]:
    """Get a user's usage statistics"""
    user = get_user(user_id)
    if not user:
        return {}
    
    today = datetime.now().date().isoformat()
    yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
    
    # Get today's scan counts
    token_scans_today = get_user_scan_count(user_id, "token_scan", today)
    wallet_scans_today = get_user_scan_count(user_id, "wallet_scan", today)
    
    # Get yesterday's scan counts
    token_scans_yesterday = get_user_scan_count(user_id, "token_scan", yesterday)
    wallet_scans_yesterday = get_user_scan_count(user_id, "wallet_scan", yesterday)
    
    # Get tracking subscriptions
    from data.database import get_user_tracking_subscriptions
    tracking_subscriptions = get_user_tracking_subscriptions(user_id)
    
    token_tracks = sum(1 for sub in tracking_subscriptions if sub.tracking_type == "token")
    wallet_tracks = sum(1 for sub in tracking_subscriptions if sub.tracking_type == "wallet")
    deployer_tracks = sum(1 for sub in tracking_subscriptions if sub.tracking_type == "deployer")
    
    return {
        "token_scans_today": token_scans_today,
        "wallet_scans_today": wallet_scans_today,
        "token_scans_yesterday": token_scans_yesterday,
        "wallet_scans_yesterday": wallet_scans_yesterday,
        "token_tracks": token_tracks,
        "wallet_tracks": wallet_tracks,
        "deployer_tracks": deployer_tracks,
        "is_premium": user.is_premium,
        "account_created": user.created_at.strftime("%d %B %Y") if user.created_at else "Unknown",
        "last_active": user.last_active.strftime("%d %B %Y %H:%M") if user.last_active else "Unknown"
    }

async def cleanup_expired_premium_subscriptions() -> int:
    """
    Clean up expired premium subscriptions
    Returns the number of subscriptions that were expired
    """
    try:
        # This function is in data.database
        cleanup_expired_premium()
        
        # Count how many were expired (would need to be implemented in database.py)
        # For now, just return a placeholder
        return 0
    except Exception as e:
        logging.error(f"Error cleaning up expired premium subscriptions: {e}")
        return 0

async def get_user_referral_code(user_id: int) -> str:
    """Get a user's referral code"""
    user = get_user(user_id)
    if not user:
        return ""
    
    # Generate a referral code if one doesn't exist
    if not hasattr(user, 'referral_code') or not user.referral_code:
        import uuid
        referral_code = str(uuid.uuid4())[:8].upper()
        
        # Save the referral code
        from data.database import update_user_referral_code
        update_user_referral_code(user_id, referral_code)
        
        return referral_code
    
    return user.referral_code

async def process_referral(referrer_id: int, referred_id: int) -> bool:
    """
    Process a referral when a new user signs up using a referral code
    Returns success status
    """
    try:
        # Check if users exist
        referrer = get_user(referrer_id)
        referred = get_user(referred_id)
        
        if not referrer or not referred:
            return False
        
        # Check if the referred user is new (created within the last day)
        if referred.created_at and (datetime.now() - referred.created_at).days > 1:
            return False
        
        # Record the referral
        from data.database import record_referral
        record_referral(referrer_id, referred_id)
        
        # Give the referrer some benefit (e.g., extra free scans or discount)
        # This would be implemented based on the referral program specifics
        
        return True
    except Exception as e:
        logging.error(f"Error processing referral from {referrer_id} to {referred_id}: {e}")
        return False

async def get_user_count_stats() -> Dict[str, int]:
    """Get user count statistics"""
    try:
        return get_user_counts()
    except Exception as e:
        logging.error(f"Error getting user count stats: {e}")
        return {
            "total_users": 0,
            "premium_users": 0,
            "active_today": 0,
            "active_week": 0,
            "active_month": 0
        }
