import pandas as pd
from langchain_core.documents import Document


def dataconveter():
    travel_data=pd.read_csv(r"C:\\Users\\harsh\\Desktop\\langchain\\data\\Reviews.csv",encoding='mac_roman')

    data=travel_data[["Location","review"]]

    travel_list = []

    # Iterate over the rows of the DataFrame
    for index, row in data.iterrows():
        # Construct an object with 'Location' and 'review' attributes
        obj = {
                'location': row['Location'],
                'review': row['review']
            }
        # Append the object to the list
        travel_list.append(obj)

        
            
    docs = []
    for entry in travel_list:
        metadata = {"location": entry['location']}
        doc = Document(page_content=entry['review'], metadata=metadata)
        docs.append(doc)
    return docs