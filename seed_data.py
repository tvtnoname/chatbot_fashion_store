import os
import time
from neo4j import GraphDatabase

# Configuration
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "fashion_password"))

def wait_for_neo4j():
    """Wait for Neo4j to be ready."""
    max_retries = 30
    for i in range(max_retries):
        try:
            driver = GraphDatabase.driver(URI, auth=AUTH)
            driver.verify_connectivity()
            print("Connected to Neo4j!")
            driver.close()
            return
        except Exception as e:
            print(f"Waiting for Neo4j... ({i+1}/{max_retries})")
            time.sleep(2)
    raise Exception("Could not connect to Neo4j.")

def setup_schema(tx):
    """Initialize constraints and indexes."""
    print("Setting up Schema...")
    queries = [
        # Constraints
        "CREATE CONSTRAINT product_id_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT style_name_unique IF NOT EXISTS FOR (s:Style) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT material_name_unique IF NOT EXISTS FOR (m:Material) REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT occasion_name_unique IF NOT EXISTS FOR (o:Occasion) REQUIRE o.name IS UNIQUE",
        "CREATE CONSTRAINT color_name_unique IF NOT EXISTS FOR (c:Color) REQUIRE c.name IS UNIQUE",
        # Indexes
        "CREATE FULLTEXT INDEX product_search IF NOT EXISTS FOR (n:Product) ON EACH [n.name, n.description]"
    ]
    for q in queries:
        tx.run(q)

def seed_data(tx):
    """Insert sample data."""
    print("Seeding Data...")
    
    # 1. Static Entities (Occasion, Material, Style, Color)
    tx.run("""
    UNWIND [
        {name: 'Beach Wedding', formality: 7},
        {name: 'Office', formality: 8},
        {name: 'Casual Street', formality: 3},
        {name: 'Gala Dinner', formality: 10}
    ] AS row
    MERGE (o:Occasion {name: row.name})
    SET o.formality_level = row.formality
    """)

    tx.run("""
    UNWIND [
        {name: 'Linen', breathability: 'High'},
        {name: 'Cotton', breathability: 'Medium'},
        {name: 'Silk', breathability: 'Medium'},
        {name: 'Wool', breathability: 'Low'}
    ] AS row
    MERGE (m:Material {name: row.name})
    SET m.breathability = row.breathability
    """)

    tx.run("""
    UNWIND ['Vintage', 'Minimalist', 'Bohemian', 'Formal'] AS name
    MERGE (:Style {name: name})
    """)
    
    # 2. Products
    cypher_products = """
    UNWIND [
        {
            id: 'P001', name: 'White Linen Shirt', price: 50.0, gender: 'Male',
            desc: 'A breathable white linen shirt perfect for summer.',
            material: 'Linen', style: 'Minimalist', occasion: 'Beach Wedding'
        },
        {
            id: 'P002', name: 'Beige Chinos', price: 60.0, gender: 'Male',
            desc: 'Comfortable beige chino pants.',
            material: 'Cotton', style: 'Formal', occasion: 'Office'
        },
        {
            id: 'P003', name: 'Floral Maxi Dress', price: 85.0, gender: 'Female',
            desc: 'Flowy floral dress.',
            material: 'Silk', style: 'Bohemian', occasion: 'Beach Wedding'
        },
        {
            id: 'P004', name: 'Navy Blue Blazer', price: 120.0, gender: 'Male',
            desc: 'Classic navy blazer.',
            material: 'Wool', style: 'Formal', occasion: 'Gala Dinner'
        }
    ] AS row
    MERGE (p:Product {id: row.id})
    SET p.name = row.name, p.price = row.price, p.gender = row.gender, p.description = row.desc
    
    // Relationships
    WITH p, row
    MATCH (m:Material {name: row.material})
    MERGE (p)-[:MADE_OF]->(m)
    
    WITH p, row
    MATCH (s:Style {name: row.style})
    MERGE (p)-[:HAS_STYLE {weight: 0.9}]->(s)
    
    WITH p, row
    MATCH (o:Occasion {name: row.occasion})
    MERGE (p)-[:SUITABLE_FOR {score: 0.9}]->(o)
    """
    tx.run(cypher_products)
    
    # 3. Complex Relationships (MATCHES_WITH)
    # Rule: White Linen Shirt matches with Beige Chinos
    tx.run("""
    MATCH (p1:Product {id: 'P001'}), (p2:Product {id: 'P002'})
    MERGE (p1)-[:MATCHES_WITH {type: 'Complementary', score: 0.95, reason: 'Classic Summer Vibe'}]->(p2)
    MERGE (p2)-[:MATCHES_WITH {type: 'Complementary', score: 0.95, reason: 'Classic Summer Vibe'}]->(p1)
    """)
    
    # Rule: Blazer matches with Chinos
    tx.run("""
    MATCH (p1:Product {id: 'P004'}), (p2:Product {id: 'P002'})
    MERGE (p1)-[:MATCHES_WITH {type: 'Complementary', score: 0.85, reason: 'Smart Casual Look'}]->(p2)
    """)

def main():
    wait_for_neo4j()
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as session:
        session.execute_write(setup_schema)
        session.execute_write(seed_data)
    driver.close()
    print("Database seeded successfully!")

if __name__ == "__main__":
    main()
