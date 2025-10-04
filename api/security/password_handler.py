﻿"""
Password hashing and verification
"""
from passlib.context import CryptContext

# Password hashing context
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    '''
    Verify a password against its hash
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password from database
        
    Returns:
        True if password matches, False otherwise
    '''
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    '''
    Hash a password
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    '''
    return pwd_context.hash(password)
