

def main():
    input_path = "/home/pkuhetu/lqs/cs336_data/owt_train.txt"
    with open(input_path, "r", encoding="utf-8") as f:
        for i in range(100):
            line = f.readline()
            print(line, end='')

if __name__ == "__main__":
    main()

