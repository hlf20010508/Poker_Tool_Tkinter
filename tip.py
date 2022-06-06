from doudizhu import Card, list_greater_cards,cards_greater

def greater(chain1,chain2):
    return cards_greater(chain1,chain2)[0]

def run(me,player1,player2):
    me = Card.card_ints_from_string(me)

    if player1!=None and player2!=None:
        player1 = Card.card_ints_from_string(player1)
        player2 = Card.card_ints_from_string(player2)
        if greater(player1,player2):
            player=player1
        else:
            player=player2
    else:
        if player1==None and player2!=None:
            player=Card.card_ints_from_string(player2)
        elif player1!=None and player2==None:
            player=Card.card_ints_from_string(player1)
        else:
            return '随意'

    result=list_greater_cards(player, me)
    types=list(result.keys())
    temp=None
    for t in types:
        for l in result[t]:
            temp=l
            break
    output=[]
    for i in temp:
        rank_int=Card.get_rank_int(i)
        r = Card.STR_RANKS[rank_int]
        output.append(r)
    return ', '.join(output)