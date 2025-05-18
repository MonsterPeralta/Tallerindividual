try:
    from src.rag import RAGSystem
    print("¡Importación exitosa!")
except ImportError as e:
    print(f"Error: {e}")