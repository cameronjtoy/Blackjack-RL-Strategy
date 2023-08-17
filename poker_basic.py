import random

def shuffle_deck(decks=1):
    # Creates a set of cards with the specified number of decks (2, 6, or 8)
    deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4 * decks
    random.shuffle(deck)
    return deck

def deal_cards(deck, player_hand, dealer_hand):
    player_hand.append(deck.pop())
    dealer_hand.append(deck.pop())
    player_hand.append(deck.pop())
    dealer_hand.append(deck.pop())

def play_round(deck, decks):
    player_hand = []
    dealer_hand = []
    deal_cards(deck, player_hand, dealer_hand)

    while sum(player_hand) < 17:
        player_hand.append(deck.pop())

    while sum(dealer_hand) <= 16:
        dealer_hand.append(deck.pop())

    # Shuffle the deck if 50% or more of the cards have been used
    if len(deck) <= (52 * decks) / 2:
        deck += shuffle_deck(decks)

    print(f"Player's hand: {player_hand}")
    return player_hand, dealer_hand

def main():
    N = 1000
    card_count = {i: 0 for i in range(2, 12)}
    decks = int(input("Choose the number of decks (2, 6, or 8): "))

    if decks not in [2, 6, 8]:
        print("Invalid number of decks. Defaulting to 2 decks.")
        decks = 2

    deck = shuffle_deck(decks)

    for _ in range(N):
        player_hand, _ = play_round(deck, decks)
        for card in player_hand:
            card_count[card] += 1

    total_cards = sum(card_count.values())
    for card, count in card_count.items():
        probability = count / total_cards
        print(f"The probability of card {card} is {probability * 100:.2f}%")

if __name__ == "__main__":
    main()
