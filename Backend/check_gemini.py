import os
import asyncio

def print_status(msg):
    print(msg)

print_status('PWD: ' + os.getcwd())
api_key_present = bool(os.getenv('GEMINI_API_KEY'))
print_status(f'GEMINI_API_KEY present: {api_key_present}')

if not api_key_present:
    print_status('Skipping live API call because GEMINI_API_KEY is not set')
    raise SystemExit(0)

try:
    import gemini_integration as gi
    print_status('gemini_integration loaded; API_KEY present in module: ' + str(bool(getattr(gi, 'API_KEY', None))))
except Exception as e:
    print_status('Failed to import gemini_integration: ' + str(e))
    raise

async def test_call():
    try:
        print_status('Making a live request to generative API (this may use credits)')
        res = await gi.generate_tips_via_gemini({'mood':'anxious'}, [{'type':'user','text':'I am anxious about my exams.'}])
        print_status('SUCCESS: got response')
        print(res)
    except Exception as e:
        print_status('LIVE CALL FAILED: ' + repr(e))

asyncio.run(test_call())
