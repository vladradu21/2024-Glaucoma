import sys
from pathlib import Path
import subprocess

import jinja2
import pandas as pd
import pdfkit

# Base directories
CURRENT_DIR = Path(__file__).parent
IMAGE_DIR = CURRENT_DIR / '../data/predict'
MASK_DIR = CURRENT_DIR / '../out/predict'
ROI_DIR = CURRENT_DIR / '../out/roi'
CSV_DIR = CURRENT_DIR / '../out/csv'
PLOTS_DIR = CURRENT_DIR / '../out/plots'
PDF_DIR = CURRENT_DIR / '../out/pdfs'
TEMPLATE_DIR = CURRENT_DIR / '../utils'


def get_csv_data(csv_path):
    df = pd.read_csv(csv_path)
    return df.columns.tolist(), df.values.tolist()


def get_file_paths(image_name):
    image_path = IMAGE_DIR / image_name
    mask_path = MASK_DIR / image_name
    roi_dir = ROI_DIR / image_name[:-4]
    plots_dir = PLOTS_DIR / image_name[:-4]
    csv_path = CSV_DIR / image_name.replace('.jpg', '.csv')

    return image_path, mask_path, roi_dir, plots_dir, csv_path


def check_directories(roi_dir, plots_dir, csv_path):
    if not roi_dir.exists():
        print(f"ROI directory {roi_dir} does not exist.")
        return False

    if not plots_dir.exists():
        print(f"Plots directory {plots_dir} does not exist.")
        return False

    if not csv_path.exists():
        print(f"CSV file {csv_path} does not exist.")
        return False

    return True


def collect_images(roi_dir, plots_dir):
    roi_images = [roi_dir / img for img in roi_dir.iterdir()]
    plots = [plots_dir / plot for plot in plots_dir.iterdir()]

    roi_images = [str(img).replace("\\", "/") for img in roi_images]
    plots = [str(plot).replace("\\", "/") for plot in plots]

    return roi_images, plots


def render_html(template_env, image_name, image_path, mask_path, roi_images, csv_headers, csv_data, plots, diagnosis):
    template = template_env.get_template('template.html')

    image_path = str(image_path).replace("\\", "/")
    mask_path = str(mask_path).replace("\\", "/")
    roi_images = [img.replace("\\", "/") for img in roi_images]
    plots = [plot.replace("\\", "/") for plot in plots]

    rendered_html = template.render(
        image_name=image_name,
        image_path=f'file:///{image_path}',
        mask_path=f'file:///{mask_path}',
        roi_images=[f'file:///{img}' for img in roi_images[::-1]],
        csv_headers=csv_headers,
        csv_data=csv_data,
        plots=[f'file:///{plot}' for plot in plots],
        diagnosis=diagnosis
    )
    return rendered_html


def save_pdf(rendered_html, pdf_path):
    config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')

    options = {
        'enable-local-file-access': None
    }

    pdfkit.from_string(rendered_html, str(pdf_path), configuration=config, options=options)

    # Open the PDF file
    subprocess.run(['start', str(pdf_path)], shell=True, check=True)


def create_pdf(image_name, diagnosis):
    image_path, mask_path, roi_dir, plots_dir, csv_path = get_file_paths(image_name)

    if not check_directories(roi_dir, plots_dir, csv_path):
        return

    roi_images, plots = collect_images(roi_dir, plots_dir)
    csv_headers, csv_data = get_csv_data(csv_path)

    template_loader = jinja2.FileSystemLoader(searchpath=str(TEMPLATE_DIR))
    template_env = jinja2.Environment(loader=template_loader)

    rendered_html = render_html(template_env, image_name, image_path, mask_path, roi_images, csv_headers, csv_data,
                                plots, diagnosis)

    pdf_path = PDF_DIR / f"{image_name.replace('.jpg', '.pdf')}"
    save_pdf(rendered_html, pdf_path)


def main():
    if len(sys.argv) != 2:
        print('Usage: python script.py <input_image_name>')
        return
    input_image_name = sys.argv[1]
    create_pdf(input_image_name, "Glaucoma")


if __name__ == '__main__':
    main()
