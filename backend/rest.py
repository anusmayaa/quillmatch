# test_model.py
import requests
import json

API_URL = "http://127.0.0.1:8000/predict"

VALIDATION_SUITE = {
    "Jane Austen": """
        It is a truth universally acknowledged that a single man in possession
        of a good fortune must be in want of a wife. However little known the
        feelings or views of such a man may be on his first entering a neighbourhood
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
        To lose one parent may be regarded as a misfortune to lose both looks like
        carelessness. I have invented an invaluable permanent invalid called Bunbury
        in order that I may be able to go down into the country whenever I choose.
        Bunbury is perfectly invaluable. If it was not for Bunbury's extraordinary
        bad health for instance I would not be able to dine with you at Willis's
        tonight for I have been really engaged to Aunt Augusta for more than a week.
        The very essence of romance is uncertainty. If ever I get married I shall
        try to forget the fact immediately.
    """,

    "Virginia Woolf": """
        Mrs Dalloway said she would buy the flowers herself. For Lucy had her
        work cut out for her. The doors would be taken off their hinges.
        Rumpelmayer's men were coming. And then thought Clarissa what a morning
        fresh as if issued to children on a beach. What a lark. What a plunge.
        For so it had always seemed to her when with a little squeak of the hinges
        which she could hear now she had burst open the French windows and plunged
        at Bourton into the open air. How fresh how calm stiller than this of course
        the air was in the early morning. Like the flap of a wave the kiss of a wave.
        Chill and sharp and yet solemn feeling as she did standing there at the open
        window that something awful was about to happen.
    """,

    "Edgar Allan Poe": """
        True nervous very very dreadfully nervous I had been and am but why will
        you say that I am mad. The disease had sharpened my senses not destroyed
        not dulled them. Above all was the sense of hearing acute. I heard all
        things in the heaven and in the earth. I heard many things in hell. How
        then am I mad. Hearken and observe how healthily how calmly I can tell
        you the whole story. It is impossible to say how first the idea entered
        my brain but once conceived it haunted me day and night. Object there was
        none. Passion there was none. I loved the old man. He had never wronged me.
        He had never given me insult. For his gold I had no desire.
    """,

    "Arthur Conan Doyle": """
        In the year 1878 I took my degree of Doctor of Medicine of the University
        of London and proceeded to Netley to go through the course prescribed for
        surgeons in the army. Having completed my studies there I was duly attached
        to the Fifth Northumberland Fusiliers as Assistant Surgeon. The regiment
        was stationed in India at the time and before I could join it the second
        Afghan war had broken out. On landing at Bombay I learned that my corps
        had advanced through the passes and was already deep in the enemy's country.
        I followed however with many other officers who were in the same situation
        as myself and succeeded in reaching Candahar in safety where I found my
        regiment and at once entered upon my new duties.
    """,
}

EDGE_CASES = {
    "Too short (should return 400)": "This text is way too short.",

    "Modern text (no correct answer — just observe)": """
        The startup had been running low on runway for months. The founding team
        gathered in the conference room to discuss their options. They could raise
        another round at a lower valuation or cut burn rate significantly. The
        CEO opened her laptop and pulled up the financial model. The numbers were
        not encouraging. They had maybe ninety days left at current spend. Someone
        suggested pivoting the product entirely. Another founder pushed back hard
        saying they were too close to product market fit to give up now. The debate
        went on for three hours with no clear resolution. Finally they agreed to
        run a structured process and get back together in a week with proposals.
    """,

    "Mixed style (Austen + Twain)": """
        It is a truth universally acknowledged that a young man from Missouri
        ain't got no use for drawing rooms and parlours and all that sort of
        foolishness. He told the truth mainly. The neighbourhood however considered
        him the rightful property of some one or other of their daughters which
        he found mighty uncomfortable I can tell you. He had no objection to hearing
        it but you want to tell me and that was invitation enough and he reckoned
        he would rather be anywhere else than sitting in that parlour listening
        to talk about who had taken Netherfield Park and what a fine thing it was.
    """,
}


def run_tests():
    print("=" * 65)
    print("  MODEL VALIDATION — ALL 7 AUTHORS")
    print("=" * 65)

    correct = 0
    total = len(VALIDATION_SUITE)
    results = []

    for expected_author, text in VALIDATION_SUITE.items():
        try:
            response = requests.post(
                API_URL,
                json={"text": text.strip()},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                predicted = data["author"]
                confidence = data["confidence"]
                all_scores = data["all_scores"]

                passed = predicted == expected_author
                if passed:
                    correct += 1

                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"\n {status} | {expected_author}")
                print(f"        Predicted : {predicted} ({confidence}%)")

                if not passed:
                    # Show top 3 scores so we can see where it's confused
                    top3 = sorted(
                        all_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    print(f"        Top 3     : {top3}")

                results.append({
                    "expected": expected_author,
                    "predicted": predicted,
                    "confidence": confidence,
                    "passed": passed
                })

            else:
                print(f"\n [!] API ERROR ({response.status_code}) for {expected_author}")
                print(f"     {response.text}")

        except requests.exceptions.ConnectionError:
            print("\n[!] Cannot connect to API. Is uvicorn running?")
            print("    Run: uvicorn main:app --reload")
            return

    # Summary
    print("\n" + "=" * 65)
    print(f"  RESULT: {correct}/{total} correct ({correct/total*100:.0f}%)")
    print("=" * 65)

    # Confidence summary table
    print("\n  Confidence scores:")
    print(f"  {'Author':<25} {'Predicted':<25} {'Conf':>6}  {'Pass'}")
    print(f"  {'-'*25} {'-'*25} {'-'*6}  {'-'*4}")
    for r in results:
        tick = "✓" if r["passed"] else "✗"
        print(f"  {r['expected']:<25} {r['predicted']:<25} {r['confidence']:>5}%  {tick}")

    # Edge cases
    print("\n" + "=" * 65)
    print("  EDGE CASES")
    print("=" * 65)

    for label, text in EDGE_CASES.items():
        try:
            response = requests.post(
                API_URL,
                json={"text": text.strip()},
                timeout=10
            )
            if response.status_code == 400:
                print(f"\n  [✓] '{label}'")
                print(f"      Correctly rejected: {response.json()['detail']}")
            elif response.status_code == 200:
                data = response.json()
                print(f"\n  [→] '{label}'")
                print(f"      Got: {data['author']} ({data['confidence']}%)")
                print(f"      Explanation: {data['explanation']}")
        except Exception as e:
            print(f"\n  [!] Error on edge case '{label}': {e}")

    print("\n" + "=" * 65)
    print("  VALIDATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()