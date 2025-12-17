"""Debug GDELT API response."""

import requests

url = "https://api.gdeltproject.org/api/v2/doc/doc"

params = {
    'query': '(AAPL OR "AAPL stock")',
    'mode': 'tonechart',  # Try tone mode
    'format': 'json',
    'startdatetime': '20240115000000',
    'enddatetime': '20240115235959',
}

print("Testing GDELT API...")
print(f"URL: {url}")
print(f"Params: {params}")
print()

response = requests.get(url, params=params, timeout=10)

print(f"Status: {response.status_code}")
print(f"Content-Type: {response.headers.get('Content-Type')}")
print(f"Response length: {len(response.text)} chars")
print()
print("First 500 chars of response:")
print(response.text[:500])
print()

if response.status_code == 200:
    try:
        data = response.json()
        print(f"Parsed JSON successfully!")
        print(f"Keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
        
        # Check for tonechart
        tonechart = data.get('tonechart', [])
        print(f"\nFound {len(tonechart)} tone bins")
        
        if tonechart:
            print("\nTone distribution:")
            total_articles = sum(bin['count'] for bin in tonechart)
            print(f"Total articles: {total_articles}")
            
            # Calculate weighted average tone
            weighted_sum = sum(bin['bin'] * bin['count'] for bin in tonechart)
            avg_tone = weighted_sum / total_articles if total_articles > 0 else 0
            print(f"Average tone: {avg_tone:.2f} (scale: -10 to +10)")
            print(f"Normalized: {avg_tone / 10:.2f} (scale: -1 to +1)")
            
            print("\nFirst few bins:")
            for bin in tonechart[:5]:
                print(f"  Tone bin {bin['bin']:+3d}: {bin['count']} articles")
                
    except Exception as e:
        print(f"JSON parse error: {e}")
