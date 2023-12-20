import summarize_format as summarize
from multiprocessing import Pool
import time
import os


def summarize_async(doc_file_path):
    try:
        summary = summarize.summarize_doc(doc_file_path)
        return doc_file_path, summary['keywords'], summary['category']
    except Exception as e:
        # Handle any exceptions that might occur during summarization
        print(f"Error processing {doc_file_path}: {str(e)}")
        return doc_file_path, None, None


def parallel_summarization(doc_file_paths: list[str]):
    with Pool() as pool:
        result = pool.map(summarize_async, doc_file_paths)
        return result


if __name__ == '__main__':

    list_files = os.listdir("./docs")

    doc_file_paths = list(map(lambda file: "./docs/" + file, list_files))
    start_time = time.perf_counter()
    result = parallel_summarization(doc_file_paths)
    finish_time = time.perf_counter()
    print("Program finished in {} seconds - using multiprocessing".format(finish_time-start_time))
    print("---")
