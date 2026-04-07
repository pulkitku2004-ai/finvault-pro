from unstructured.partition.pdf import partition_pdf

def parse_pdf(path):
    elements = partition_pdf(
        filename = path,
        strategy = 'hi_res',
        infer_table_structure=True,
        chunking_strategy="by_title", # This helps group text logically
        max_characters=1000,
        combine_text_under_n_chars=200
    )

    texts=[]

    for element in elements:
        texts.append(str(element))

    return texts

if __name__ == "__main__":
    data=parse_pdf("data/hdfc_q3.pdf")
    print(len(data))
    print(data[:5])