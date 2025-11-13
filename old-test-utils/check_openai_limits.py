"""
Check your OpenAI API rate limits and account tier.
This helps determine the optimal batch size for concurrent processing.
"""

import os
import sys
from openai import OpenAI
from openai import RateLimitError, APIError

def check_rate_limits():
    """Check OpenAI API rate limits and account information."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("\nTo set it:")
        print("  Windows: set OPENAI_API_KEY=your-key-here")
        print("  Linux/Mac: export OPENAI_API_KEY=your-key-here")
        return
    
    print("="*60)
    print("OpenAI API Rate Limit Checker")
    print("="*60)
    print()
    
    client = OpenAI(api_key=api_key)
    
    # Try a simple request to check rate limits
    print("Testing API access and checking rate limits...")
    print()
    
    try:
        # Make a simple, cheap request to test
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Say 'test'"}
            ],
            max_tokens=5
        )
        
        print("✅ API connection successful!")
        print()
        
        # Check response headers for rate limit info
        # Note: OpenAI Python SDK doesn't expose all headers directly
        # We'll need to infer from behavior
        
        print("Rate Limit Information:")
        print("-" * 60)
        print()
        print("⚠️  Note: OpenAI doesn't expose exact rate limits via API.")
        print("   We'll test with a small batch to determine your limits.")
        print()
        
        # Try to determine tier by making multiple requests
        print("Testing concurrent request capability...")
        print()
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        def make_test_request(i):
            """Make a test request."""
            try:
                start = time.time()
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": f"Say '{i}'"}
                    ],
                    max_tokens=5
                )
                elapsed = time.time() - start
                return (i, True, elapsed, None)
            except RateLimitError as e:
                return (i, False, 0, "RateLimitError")
            except APIError as e:
                return (i, False, 0, f"APIError: {str(e)}")
            except Exception as e:
                return (i, False, 0, f"Error: {str(e)}")
        
        # Test with different batch sizes
        test_sizes = [1, 3, 5, 10]
        successful_sizes = []
        
        for batch_size in test_sizes:
            print(f"  Testing {batch_size} concurrent requests...", end=" ")
            
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [executor.submit(make_test_request, i) for i in range(batch_size)]
                results = []
                errors = []
                
                for future in as_completed(futures):
                    result = future.result()
                    if result[1]:  # Success
                        results.append(result)
                    else:
                        errors.append(result)
            
            elapsed = time.time() - start_time
            
            if len(errors) == 0:
                print(f"✅ Success ({elapsed:.2f}s)")
                successful_sizes.append(batch_size)
            else:
                print(f"❌ Failed ({len(errors)} errors)")
                if "RateLimitError" in str(errors[0][3]):
                    print(f"     → Rate limit hit at {batch_size} concurrent requests")
                break
        
        print()
        print("="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        print()
        
        if len(successful_sizes) == 0:
            print("❌ No successful concurrent requests.")
            print("   You may be on the free tier with very strict limits.")
            print("   Recommended batch size: 1 (sequential processing)")
        elif max(successful_sizes) < 3:
            print("✅ You can handle 1-2 concurrent requests.")
            print("   You're likely on the FREE TIER.")
            print()
            print("   Recommended batch size: 1-2")
            print("   Note: Free tier has 3 requests/minute limit")
        elif max(successful_sizes) < 10:
            print("✅ You can handle 3-5 concurrent requests.")
            print("   You're likely on TIER 1 ($5+ spent).")
            print()
            print("   Recommended batch size: 3-5")
            print("   Tier 1: 500 requests/minute")
        elif max(successful_sizes) < 20:
            print("✅ You can handle 5-10 concurrent requests.")
            print("   You're likely on TIER 2 ($50+ spent).")
            print()
            print("   Recommended batch size: 5-10")
            print("   Tier 2: 5,000 requests/minute")
        else:
            print("✅ You can handle 10+ concurrent requests.")
            print("   You're likely on TIER 3 ($500+ spent).")
            print()
            print("   Recommended batch size: 10-20")
            print("   Tier 3: 10,000 requests/minute")
        
        print()
        print("="*60)
        print("ACCOUNT INFORMATION")
        print("="*60)
        print()
        print("To check your exact tier and limits:")
        print("  1. Go to: https://platform.openai.com/account/limits")
        print("  2. Check 'Rate limits' section")
        print("  3. Look for 'Requests per minute' for chat completions")
        print()
        print("Tier Reference:")
        print("  Free tier: 3 requests/minute")
        print("  Tier 1: 500 requests/minute")
        print("  Tier 2: 5,000 requests/minute")
        print("  Tier 3: 10,000 requests/minute")
        print()
        
        # Check usage/credits
        print("To check your API credits/usage:")
        print("  1. Go to: https://platform.openai.com/usage")
        print("  2. View your current usage and remaining credits")
        print()
        
    except RateLimitError as e:
        print("❌ Rate limit error detected!")
        print(f"   Error: {str(e)}")
        print()
        print("You're likely on the FREE TIER with strict limits.")
        print("Recommended: Use batch_size=1 (sequential processing)")
        print()
    except APIError as e:
        print(f"❌ API Error: {str(e)}")
        print()
        print("Possible issues:")
        print("  - Invalid API key")
        print("  - Insufficient credits")
        print("  - Account restrictions")
        print()
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print()
        print("Please check:")
        print("  1. Your API key is valid")
        print("  2. You have credits in your account")
        print("  3. Your account has access to gpt-4o-mini")
        print()


if __name__ == "__main__":
    check_rate_limits()

