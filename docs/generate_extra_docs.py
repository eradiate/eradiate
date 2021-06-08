from pathlib import Path

from eradiate._config import EradiateConfig


def format_help_dicts(help_dicts, display_defaults=False):
    help_strs = []
    for help_dict in help_dicts:
        help_str = f"{help_dict['var_name']} ({'Required' if help_dict['required'] else 'Optional'}"

        if help_dict.get("default") and display_defaults:
            help_str += f", Default={help_dict['default']})"
        else:
            help_str += ")"

        if help_dict.get("help_str"):
            help_str += f"\n    {help_dict['help_str']}"
        help_strs.append(help_str)

    return "\n\n".join(help_strs)


def main():
    path = Path("rst/reference/generated/env_vars.rst").absolute()
    print(f"Writing config variables to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("Environment variables\n---------------------\n\n")
        f.write(EradiateConfig.generate_help(formatter=format_help_dicts))


if __name__ == "__main__":
    main()
