from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from requests import post
from sys import exit


if __name__=="__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "x", action="store", help="X coordinate of the point at which the function value is requested."
    )
    parser.add_argument(
        "y", action="store", help="Y coordinate of the point at which the function value is requested."
    )
    varss = vars(parser.parse_args())
    X = varss["x"] + " " + varss["y"]

    POST_args = {"secret_key": "WVRjjVBm", "x": X, "type": "small"}
    response = post("http://optimize-me.ddns.net:8080/", data=POST_args, timeout=1.5)

    try:
        print(isinstance(response.content, float))
        Y = str(response.content, encoding="utf8")
    except:
        exit()

    with open("fvals.txt", "a", encoding="utf8") as file:
        file.write(X + "|" + Y + "\n")
