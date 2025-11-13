"""x402 payment scheme definitions"""
from enum import Enum

class PaymentScheme(Enum):
    EIP3009 = "eip3009"      # Gasless transfers
    SOLANA_SPL = "solana_spl" # Solana token transfers
    DIRECT = "direct"         # Direct transfers

class Network(Enum):
    BASE_MAINNET = "base"
    BASE_SEPOLIA = "base-sepolia" 
    SOLANA_MAINNET = "solana"
    SOLANA_DEVNET = "solana-devnet"