## Homework 3: blackbox optimization

In this task you will optimize a true black box function $F$, calculation if which is relatively expensive. The good news are that you have access to another **trial function** $f$, which is much cheaper evaluationwise. This is reminiscent of the situation, when one needs to choose the training hyperparameters of some large model: you can try to study the effect of these hyperparameters on a smaller model, and you can make some trial runs. 

Your final goal is to find the largest possible value of the blackbox function as possible while minimizing the number of calculations. You have two types of actions: calculate $f(x)$ or $F(x)$. To be concrete, if you  calculated $f(x_1) \dots f(x_n)$ and $F(X_1)\dots F(X_N)$, then your final score would be 
$$
score = \max\limits_{1 \leq i \leq N}F(X_i) - CN - cn
$$
with constant $C$ and $c$ equal to $1$ and $0.01$ respectively.

The only thing you know about these functions is that both are defined over $\left[-1, 1 \right]^2$, and they are *somewhat smooth*. 

### How to calculate functions

Both large and small functions are available at http://optimize-me.ddns.net:8080/, and to access its values you need to send a POST request with three fields: `secret_key`, `x` and `type`. For example:

```bash
curl -X POST http://optimize-me.ddns.net:8080/ -d "secret_key=$MY_KEY" -d "x=-0.5 -0.5" -d "type=small"
```

1. The `secret_key` is your identifier. To obtain it, generate an arbitrary string of utf8 characters and DM it to me. You would be able to make submissions only after I add your identifier to the list of allowed users.
2. The `x` field must contain two float numbers separated by single whitespace -- the coordinates of point to query.
3. The `type` can be either `small` or `large`: the type of a function to evaluate.
4. The response contains either a single float (value of the function at `x`) or an error string.

#### The rules
1. Have fun and don't cheat. 
2. Allowed modules: `numpy`, `pandas`, `scipy`, `requests` and their dependencies. 
2. The maximum score is 10. You get 4, 6, 8, 10 stars for achieving the final score of 50, 90, 130 and 160 respectively.
3. If you achieved the highest score among all participants, you get **two additional stars**. If you are in top-5, you get **one additional star**.
4. **You are limited to one request per second.** More frequent responses would be responded with an error string, but **counted towards your number of trials**.
5. There are plenty of different black-box optimization schemes. If you do something beyond applying scipy.optimize and what we did in class, I will invite you to give a 5-10 minutes talk on one of the following practical lessons to earn one more star. 
6. The server **WILL** die occasionally (especially when the deadline is near), your code should not fail in this case. If it doesn't respond to you for more than 2 hours, then DM me.
7. If you found a bug or your requests do not work for some reason -- DM me.
8. Deadline: 20.03.2025.
