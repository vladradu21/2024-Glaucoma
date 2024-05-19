import os
import sys

import jinja2
import pandas as pd
import pdfkit

# Hyperparameters etc.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, '../data/predict/')
MASK_DIR = os.path.join(BASE_DIR, '../out/predict/')
ROI_DIR = os.path.join(BASE_DIR, '../out/roi/')
CSV_DIR = os.path.join(BASE_DIR, '../out/csv/')
PLOTS_DIR = os.path.join(BASE_DIR, '../out/plots/')
PDF_DIR = os.path.join(BASE_DIR, '../out/pdfs/')


def get_csv_data(csv_path):
    df = pd.read_csv(csv_path)
    return df.columns.tolist(), df.values.tolist()


def get_file_paths(image_name):
    image_path = os.path.join(IMAGE_DIR, image_name)
    mask_path = os.path.join(MASK_DIR, image_name)
    roi_dir = os.path.join(ROI_DIR, image_name[:-4])
    plots_dir = os.path.join(PLOTS_DIR, image_name[:-4])
    csv_path = os.path.join(CSV_DIR, image_name.replace('.jpg', '.csv'))

    return image_path, mask_path, roi_dir, plots_dir, csv_path


def check_directories(roi_dir, plots_dir, csv_path):
    if not os.path.exists(roi_dir):
        print(f"ROI directory {roi_dir} does not exist.")
        return False

    if not os.path.exists(plots_dir):
        print(f"Plots directory {plots_dir} does not exist.")
        return False

    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} does not exist.")
        return False

    return True


def collect_images(roi_dir, plots_dir):
    roi_images = [os.path.join(roi_dir, img) for img in os.listdir(roi_dir)]
    plots = [os.path.join(plots_dir, plot) for plot in os.listdir(plots_dir)]

    roi_images = [img.replace("\\", "/") for img in roi_images]
    plots = [plot.replace("\\", "/") for plot in plots]

    return roi_images, plots


def render_html(template_env, image_name, image_path, mask_path, roi_images, csv_headers, csv_data, plots, diagnosis):
    template = template_env.get_template('template.html')

    image_path = image_path.replace("\\", "/")
    mask_path = mask_path.replace("\\", "/")
    roi_images = [img.replace("\\", "/") for img in roi_images]
    plots = [plot.replace("\\", "/") for plot in plots]

    rendered_html = template.render(
        image_name=image_name,
        image_path=f'file:///{image_path}',
        mask_path=f'file:///{mask_path}',
        roi_images=[f'file:///{img}' for img in roi_images],
        csv_headers=csv_headers,
        csv_data=csv_data,
        plots=[f'file:///{plot}' for plot in plots[:-1]],
        diagnosis=diagnosis
    )
    return rendered_html


def save_pdf(rendered_html, pdf_path):
    config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')

    options = {
        'enable-local-file-access': None
    }

    pdfkit.from_string(rendered_html, pdf_path, configuration=config, options=options)


def create_pdf(image_name, diagnosis):
    image_path, mask_path, roi_dir, plots_dir, csv_path = get_file_paths(image_name)

    if not check_directories(roi_dir, plots_dir, csv_path):
        return

    roi_images, plots = collect_images(roi_dir, plots_dir)
    csv_headers, csv_data = get_csv_data(csv_path)

    template_loader = jinja2.FileSystemLoader(searchpath="./")
    template_env = jinja2.Environment(loader=template_loader)

    rendered_html = render_html(template_env, image_name, image_path, mask_path, roi_images, csv_headers, csv_data,
                                plots, diagnosis)

    pdf_path = os.path.join(PDF_DIR, f"{image_name.replace('.jpg', '.pdf')}")
    save_pdf(rendered_html, pdf_path)


def main():
    if len(sys.argv) != 2:
        print('Usage: python script.py <input_image_name>')
        return
    input_image_name = sys.argv[1]
    create_pdf(input_image_name, "Glaucoma")


if __name__ == '__main__':
    main()
