
from trueskill import Rating, rate_1vs1
 
# 初始化两名玩家的评分
player1 = Rating()
player2 = Rating()
 
# 模拟比赛结果：player1 获胜
player1, player2 = rate_1vs1(player1, player2)
player2, player1 = rate_1vs1(player2, player1)
 
print(f"Player 1 的新评分: {player1}")
print(f"Player 2 的新评分: {player2}")