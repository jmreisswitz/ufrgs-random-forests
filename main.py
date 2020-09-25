import argparse


def get_ntree_from_argparser():
    parser = argparse.ArgumentParser(description='Random Forests')
    parser.add_argument('--ntree', type=int, default=20, help='Number of random forests on ensemble.')
    args = parser.parse_args()
    return args.ntree


def main():
    print(get_ntree_from_argparser())


if __name__ == '__main__':
    main()
