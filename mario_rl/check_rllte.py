import inspect

import rllte
from rllte.xplore.reward import ICM, RE3, RND


def main():
    print("rllte=ok")
    print(f"rllte_path={inspect.getfile(rllte)}")
    print(f"rnd_class={RND.__module__}.{RND.__name__}")
    print(f"icm_class={ICM.__module__}.{ICM.__name__}")
    print(f"re3_class={RE3.__module__}.{RE3.__name__}")


if __name__ == "__main__":
    main()
