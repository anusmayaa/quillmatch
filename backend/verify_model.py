import requests
import json

# Ensure your FastAPI server is running on this port before executing!
API_URL = "http://127.0.0.1:8000/predict"

import requests
import json

# Updated to your live port
API_URL = "http://127.0.0.1:8000/predict"

VALIDATION_SUITE = {
    

    "Jane Austen": """
        It is a truth universally acknowledged that a single man in possession
        of a good fortune must be in want of a wife. However little known the
        feelings or views of such a man may be on his first entering a neighbourhood,
        this truth is so well fixed in the minds of the surrounding families that
        he is considered as the rightful property of some one or other of their
        daughters. My dear Mr Bennet said his lady to him one day have you heard
        that Netherfield Park is let at last. Mr Bennet replied that he had not.
        But it is returned she for Mrs Long has just been here and she told me
        all about it. Mr Bennet made no answer. Do not you want to know who has
        taken it cried his wife impatiently. You want to tell me and I have no
        objection to hearing it. This was invitation enough.
    """,

    "Charles Dickens": """
        Marley was dead to begin with. There is no doubt whatever about that.
        The register of his burial was signed by the clergyman the clerk the
        undertaker and the chief mourner. Scrooge signed it. And Scrooge's name
        was good upon Change for anything he chose to put his hand to. Old Marley
        was as dead as a door nail. Mind I do not mean to say that I know of my
        own knowledge what there is particularly dead about a door nail. I might
        have been inclined myself to regard a coffin nail as the deadest piece
        of ironmongery in the trade. But the wisdom of our ancestors is in the
        simile and my unhallowed hands shall not disturb it or the country's done
        for. You will therefore permit me to repeat emphatically that Marley was
        dead as a door nail.
    """,

    "Mark Twain": """
        You don't know about me without you have read a book by the name of
        The Adventures of Tom Sawyer but that ain't no matter. That book was
        made by Mr Mark Twain and he told the truth mainly. There was things
        which he stretched but mainly he told the truth. That is nothing. I never
        seen anybody but lied one time or another without it was Aunt Polly or
        the widow or maybe Mary. Aunt Polly she is and Mary and the Widow Douglas
        is all told about in that book which is mostly a true book with some
        stretchers as I said before. Now the way that the book winds up is this.
        Tom and me found the money that the robbers hid in the cave and it made
        us rich. We got six thousand dollars apiece all gold.
    """,

    "Oscar Wilde": """
        The truth is rarely pure and never simple. Modern life would be very
        tedious if it were either and modern literature a complete impossibility.
        The good ended happily and the bad unhappily. That is what fiction means.
        To lose one parent may be regarded as a misfortune to lose both looks like
        carelessness. I have invented an invaluable permanent invalid called Bunbury
        in order that I may be able to go down into the country whenever I choose.
        Bunbury is perfectly invaluable. If it was not for Bunbury's extraordinary
        bad health for instance I would not be able to dine with you at Willis's
        tonight for I have been really engaged to Aunt Augusta for more than a week.
        The very essence of romance is uncertainty. If ever I get married I shall
        try to forget the fact immediately.
    """
}

def run_system_check():
    print("=" * 65)
    print("   🔍 RUNNING PRODUCTION MODEL TRUTH VALIDATION")
    print("=" * 65)
    
    correct = 0
    total = len(VALIDATION_SUITE)
    
    for expected_author, text in VALIDATION_SUITE.items():
        payload = {"text": text.strip()}
        
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                predicted_author = data.get("author")
                confidence = data.get("confidence")
                
                if predicted_author == expected_author:
                    print(f" [✓] PASS | Expected: {expected_author:<20} -> Got: {predicted_author:<20} ({confidence}%)")
                    correct += 1
                else:
                    print(f" [✗] FAIL | Expected: {expected_author:<20} -> Got: {predicted_author:<20} ({confidence}%)")
                    print(f"     ↳ Top scores: {data.get('all_scores')}")
            else:
                print(f" [!] API Error ({response.status_code}) for {expected_author}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("[!] Critical Error: Could not connect to the API. Is your main.py server running?")
            return
            
    print("=" * 65)
    accuracy = (correct / total) * 100
    print(f" SYSTEM VERIFICATION COMPLETE: {correct}/{total} Correct ({accuracy:.1%})")
    print("=" * 65)

if __name__ == "__main__":
    run_system_check()