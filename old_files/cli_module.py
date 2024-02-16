import argparse

def main():
    parser = argparse.ArgumentParser(description="Example CLI with an integer argument.")
    parser.add_argument('number', type=int, help='An integer argument')

    args = parser.parse_args()
    result = perform_operation(args.number)
    print(f"Result of the operation: {result}")

def perform_operation(number):
    # You can perform your computation or operation here
    return number * 2

if __name__ == "__main__":
    main()
