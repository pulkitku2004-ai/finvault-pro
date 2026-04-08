from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
username = "neo4j"
password= "password"

driver = GraphDatabase.driver(uri, auth=(username, password))

def get_executives_for_period(period):
    with driver.session() as session:

        result= session.run(
            """
            MATCH (e:Executive)-[:SPOKE_IN] -> (p:FIN_PERIOD {name:$period})
            RETURN e.name
            """,
            period= period
        )
        return [record["e.name"] for record in result]
    
if __name__ == "__main__":
    executives = get_executives_for_period("Q3 FY2025")

    print("Executives speaking in Q3 FY2025:")
    print(executives)

    #after this the system can do vector search and graph traversal

#Right now i have : PDF -> Vector RAg, PDF -> Knowledge Graph, after this we will make them talk to each other.
