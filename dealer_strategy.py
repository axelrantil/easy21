from env import step

def dealer_strategy(s):
    """Player uses same strategy as dealer
    Used for testing purposes
    """
    i = 1
    print("Start cards")
    print("Dealer hand: " + str(s[0]))
    print("Player hand: " + str(s[1]))
    while 0 < s[1] < 17:
        s, r = step(s, 1)  # Hit
        print("Draw " + str(i))
        i += 1
        print("Dealer hand: " + str(s[0]))
        print("Player hand: " + str(s[1]))
    if r == 0:
        s, r = step(s, 0)  # Stick
    if r == 1:
        print("Win!")
    elif r == 0:
        print("Draw")
    else:
        print("Loss")