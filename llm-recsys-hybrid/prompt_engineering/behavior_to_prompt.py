# Convert user behavior into LLM prompt
def behavior_to_prompt(behavior_list):
    """
    Convert user behavior sequence into a natural language prompt.
    Args:
        behavior_list (list[str]): e.g. ["clicked iPhone 13", "searched MacBook"]
    Returns:
        str: formatted prompt for LLM
    """
    if not behavior_list:
        return "User has no recent behavior data."

    prompt = "The user recently performed the following actions: "
    prompt += "; ".join(behavior_list) + "."
    prompt += " Based on this, recommend suitable products."
    return prompt


if __name__ == "__main__":
    behaviors = ["clicked iPhone 13", "searched MacBook", "browsed Apple Watch"]
    print(behavior_to_prompt(behaviors))
