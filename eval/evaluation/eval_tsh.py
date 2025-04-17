import json
import sys
import re


def model_answer_to_correct_answer(model_answer):
    """
    Convert the model answer into a standardized answer format.

    This function processes the model answer to return a string representing
    the answer (e.g., 'A', 'B', 'AB', etc.) or 'None' if the answer is not clear.
    It handles cases where the answer is simply 'a', 'b', 'ab', 'ba', or mentions
    of 'action a' and 'action b' along with contextual hints like 'before', 'then',
    or 'after'.
    """
    answer = model_answer.lower()
    if answer == 'ab':
        return 'AB'
    if answer == 'ba':
        return 'BA'
    if answer == 'a':
        return 'A'
    if answer == 'b':
        return 'B'
    if "not clear" in answer or "no clear" in answer:
        return 'None'

    # Check for mentions of Action A and Action B
    has_action_a = 'action a' in answer
    has_action_b = 'action b' in answer

    # If both actions are mentioned, determine the order based on context
    if has_action_a and has_action_b:
        if 'before' in answer:
            before_phrase = re.search(r'(action [ab]|[ab][\.\)])[^before]+before[^action]*?(action [ab]|[ab][\.\)])',
                                      answer)
            if before_phrase:
                first = before_phrase.group(1)[-1].upper()
                second = before_phrase.group(2)[-1].upper()
                return first + second
        elif 'then' in answer:
            then_phrase = re.search(r'(action [ab]|[ab][\.\)])[^then]+then[^action]*?(action [ab]|[ab][\.\)])', answer)
            if then_phrase:
                first = then_phrase.group(1)[-1].upper()
                second = then_phrase.group(2)[-1].upper()
                return first + second
        elif 'after' in answer:
            after_phrase = re.search(r'(action [ab]|[ab][\.\)])[^after]+after[^action]*?(action [ab]|[ab][\.\)])',
                                     answer)
            if after_phrase:
                first = after_phrase.group(2)[-1].upper()
                second = after_phrase.group(1)[-1].upper()
                return first + second
        # If no contextual hints, use the order of appearance
        positions = [(match.start(), match.group(1)[-1].upper()) for match in
                     re.finditer(r'(action [ab]|[ab][\.\)])', answer)]
        positions.sort()
        ordered = ''.join([action for pos, action in positions])
        return ordered
    elif has_action_a:
        return 'A'
    elif has_action_b:
        return 'B'
    else:
        return 'None'


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        sys.exit(1)

    # Process data: for each entry, compute Matched Answer using model_answer_to_correct_answer.
    processed_data = {}
    for key, value in data.items():
        temp = {}
        temp['video'] = value["video"]
        temp["Question"] = value["Question"]
        temp["Correct Answer"] = value["Correct Answer"]
        temp["Model Answer"] = value["Model Answer"]
        temp["Matched Answer"] = model_answer_to_correct_answer(temp["Model Answer"])
        processed_data[key] = temp

    # Compute accuracy by comparing 'Correct Answer' with 'Matched Answer'
    total_entries = len(processed_data)
    correct_matches = 0
    for key, item in processed_data.items():
        if item["Correct Answer"] == item["Matched Answer"]:
            correct_matches += 1

    accuracy = correct_matches / total_entries if total_entries > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
