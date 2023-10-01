def format_code_response(response_text: str) -> str:
    """

    :param response_text:
    :return:
    """
    # List of supported languages
    languages = ["javascript", "python", "java", "csharp", "ruby", "php", "go", "rust", "typescript", "swift", "html", "css"]

    # Loop through each language to find and replace
    for lang in languages:
        if f"```{lang}" in response_text.lower():
            response_text = response_text.replace(f"```{lang}", "```", 1)
            break

    return response_text
