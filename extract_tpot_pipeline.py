# function to extract only pipeline from exported tpot py script


def extract_pipeline(filename: str = None, lines_to_remove: list = ['']):
    # check for existance of filename
    if filename is None:
        print("Please provide filename")

    elif filename is not None:
        print("Extracting pipeline from file: ", filename)

        if lines_to_remove != '':
            print("Removing lines starting with:")
            print(lines_to_remove)
            with open(filename) as oldfile, open('tpot_pipeline.py', 'w') as newfile:
                for line in oldfile:
                    if not any(bad_word in line for bad_word in lines_to_remove):
                        newfile.write(line)
            import tpot_pipeline
        elif lines_to_remove == '':
            print("Please provide starting string")
    return 0


