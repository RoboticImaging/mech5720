import sys
import matplotlib.pyplot as plt
import rawpy


def main():
    if len(sys.argv) < 2:
        print("Usage: python view_raw.py <FILENAME>")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        # Load and view the raw bayer image
        with rawpy.imread(filename) as raw:

            rgb = raw.raw_image

        # Display using Matplotlib
        plt.figure(figsize=(10, 6))
        plt.imshow(rgb)
        plt.axis("off")  # Hide coordinate axes
        plt.title(filename)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()