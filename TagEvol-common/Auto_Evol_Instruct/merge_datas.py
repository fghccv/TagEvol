import json
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Evolutionary Instructions Generation")
    parser.add_argument('--data1_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--data2_path', type=str, required=True, help='Path to the source JSON file')
    parser.add_argument('--target_path', type=str, required=True, help='Path to the target JSON file')
    
    args = parser.parse_args()
    datas1 = json.load(open(args.data1_path))
    datas2 = json.load(open(args.data2_path))
    datas = datas1 + datas2
    print(len(datas))
    json.dump(datas, open(args.target_path, 'w'))
