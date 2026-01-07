from src.fashion_store.main import run_fashion_consultant

if __name__ == "__main__":
    import sys
    query = "Phối cho tôi một bộ đồ Streetwear thật ngầu để đi chơi tối nay với bạn bè"
    if len(sys.argv) > 1:
        query = sys.argv[1]
    
    run_fashion_consultant(query)
