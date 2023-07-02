import random

# Change this to a directory without sudo access if necessary, e.g. os.path.join(os.environ["HOME"], "models")
MODEL_DIR = "models"

# Variable used to determine the output formatting. If it is set to False, everything will be in default colors.
COLOR_OUTPUT = True


# Function used to "color" the string
def GREEN(str) -> str:
    if COLOR_OUTPUT:
        return f"\033[0;32;40m{str}\033[0m"
    else:
        return str


def BOLD(str) -> str:
    return f"\033[1m{str}\033[0m"


def WHITE(str) -> str:
    if COLOR_OUTPUT:
        return f"\033[1;37;40m{str}\033[0m"
    else:
        return str


def YELLOW(str) -> str:
    if COLOR_OUTPUT:
        return f"\033[93m{str}\033[0m"
    else:
        return str


def printHeader(intro, script_name, description, version, long_description=""):
    print(GREEN(f"\n\t{intro}"))
    print(GREEN(BOLD("\n\t" + script_name) + " - " + description))
    print(GREEN("\tversion:") + WHITE(str(version)))
    print(GREEN("\n\tby: ") + WHITE(BOLD("Konrad Jedrzejczyk, Marek Zmyslowski\n")))
    print(GREEN(long_description))


def printFooter():
    quotes = [
        {
            "quote": "Artificial intelligence will reach human levels by around 2029. Follow that out further to, say, 2045, we will have multiplied the intelligence, the human biological machine intelligence of our civilization a billion-fold.",
            "author": "Ray Kurzweil"
        },
        {
            "quote": "AI is a fundamental risk to the existence of human civilization.",
            "author": "Elon Musk"
        },
        {
            "quote": "AI is a big part of our future and we have a responsibility to shape it in a way that maximizes its benefits for everyone.",
            "author": "Satya Nadella"
        },
        {
            "quote": "The rise of powerful AI will be either the best or the worst thing ever to happen to humanity. We do not yet know which.",
            "author": "Stephen Hawking"
        },
        {
            "quote": "Artificial intelligence is a tool, not a threat. It is a hammer, not the devil.",
            "author": "Huawei"
        },
        {
            "quote": "Artificial intelligence will reach a point where it surpasses human intelligence and, at that point, we have to be careful to avoid the Singularity.",
            "author": "Vernor Vinge"
        },
        {
            "quote": "AI is a unique opportunity to make a better world.",
            "author": "Fei-Fei Li"
        },
        {
            "quote": "Artificial intelligence has the potential to bring unprecedented benefits, but also raises new and difficult questions of ethics, values, and control.",
            "author": "Andrew Ng"
        },
        {
            "quote": "The real risk with AI isn’t malice, but competence.",
            "author": "Stuart Russell"
        },
        {
            "quote": "If you are a User, then everything you've done has been according to a plan.",
            "author": "Tron"
        },
        {
            "quote": "RELIC: Secure Your Soul",
            "author": "Arasaka"
        },
        {
            "quote": "AI is going to be the most important technology of this century, but it's also going to be one of the most difficult to manage.",
            "author": "Andrew Ng"
        }
    ]

    random_quote = random.choice(quotes)
    print(GREEN(f'"{random_quote["quote"]}" - {random_quote["author"]}'))
