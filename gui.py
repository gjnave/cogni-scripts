import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import subprocess
import os
import platform
import random
import string

class RembgGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("rembg GUI")
        self.geometry("1200x900")

        # Define extra parameters options
        self.extra_params_options = {
            "Return only the mask": "-om",
            "Apply alpha matting": "-a"
        }

        self.create_widgets()

    def create_widgets(self):
        self.center_window()

        tk.Label(self, text="Select Function").grid(row=0, column=0, pady=10, padx=10)

        self.function_var = tk.StringVar(value="Single Image")
        function_menu = tk.OptionMenu(self, self.function_var, "Single Image", "Folder", "Web Image", command=self.update_inputs)
        function_menu.grid(row=0, column=1, pady=10, padx=10)
        self.function_var.trace_add("write", self.update_inputs)

        self.input_label = tk.Label(self, text="Input Path")
        self.input_label.grid(row=1, column=0, pady=10, padx=10)

        self.input_entry = tk.Entry(self, width=80)
        self.input_entry.grid(row=1, column=1, pady=10, padx=10)

        self.input_button = tk.Button(self, text="Browse", command=self.browse_input)
        self.input_button.grid(row=1, column=2, pady=10, padx=10)

        self.output_label = tk.Label(self, text="Output Path")
        self.output_label.grid(row=2, column=0, pady=10, padx=10)

        self.output_entry = tk.Entry(self, width=80)
        self.output_entry.grid(row=2, column=1, pady=10, padx=10)

        self.output_button = tk.Button(self, text="Browse", command=self.browse_output)
        self.output_button.grid(row=2, column=2, pady=10, padx=10)

        self.extra_params_label = tk.Label(self, text="Extra Options (Select from below)")
        self.extra_params_label.grid(row=3, column=0, pady=10, padx=10)

        self.extra_params_list = tk.Listbox(self, height=8, selectmode=tk.MULTIPLE, width=100)
        for option in self.extra_params_options.keys():
            self.extra_params_list.insert(tk.END, option)
        self.extra_params_list.grid(row=4, column=0, columnspan=3, pady=10, padx=10)

        self.submit_button = tk.Button(self, text="Submit", command=self.run_rembg)
        self.submit_button.grid(row=5, column=0, columnspan=3, pady=20)

        self.result_label = tk.Label(self, text="", fg="green")
        self.result_label.grid(row=6, column=0, columnspan=3)

        self.image_label = tk.Label(self)
        self.image_label.grid(row=7, column=0, columnspan=3, pady=10, padx=10)

        self.open_output_button = tk.Button(self, text="Open Output Folder", command=self.open_output_folder)
        self.open_output_button.grid(row=8, column=0, columnspan=3, pady=20)

        self.update_inputs()

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def update_inputs(self, *args):
        function = self.function_var.get()
        if function == "Single Image":
            self.input_label.config(text="Input File")
            self.output_label.config(text="Output File")
            self.input_entry.delete(0, tk.END)
            self.output_entry.delete(0, tk.END)
            self.image_label.grid()
        elif function == "Folder":
            self.input_label.config(text="Input Folder")
            self.output_label.config(text="Output Folder")
            self.input_entry.delete(0, tk.END)
            self.output_entry.delete(0, tk.END)
            self.image_label.grid_remove()
        elif function == "Web Image":
            self.input_label.config(text="Image URL")
            self.output_label.config(text="Output File")
            self.input_entry.delete(0, tk.END)
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, self.generate_random_filename() + ".jpg")
            self.image_label.grid()
        else:
            self.input_label.config(text="Input Path")
            self.output_label.config(text="Output Path")
            self.input_entry.delete(0, tk.END)
            self.output_entry.delete(0, tk.END)
            self.image_label.grid_remove()

    def browse_input(self):
        function = self.function_var.get()
        if function == "Single Image":
            path = filedialog.askopenfilename(title="Select Input File")
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, path)
            self.set_default_output(path)
        elif function == "Folder":
            path = filedialog.askdirectory(title="Select Input Folder")
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, path)
            self.set_default_output(path)
        else:
            messagebox.showinfo("Info", "Please manually enter the input parameters.")

    def browse_output(self):
        function = self.function_var.get()
        if function == "Single Image":
            path = filedialog.asksaveasfilename(title="Select Output File")
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)
        elif function == "Folder":
            path = filedialog.askdirectory(title="Select Output Folder")
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)
        else:
            messagebox.showinfo("Info", "Please manually enter the output parameters.")

    def set_default_output(self, input_path):
        if input_path:
            base_name = os.path.basename(input_path)
            name, ext = os.path.splitext(base_name)
            output_name = f"{name}_rembg{ext}"
            output_path = os.path.join(os.getcwd(), output_name)
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, output_path)

    def generate_random_filename(self):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    def display_image(self, path):
        try:
            image = Image.open(path)
            image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
        except Exception as e:
            self.image_label.config(text="Unable to display image.")
            self.image_label.image = None

    def run_rembg(self):
        function_mapping = {
            "Single Image": "i",
            "Folder": "p",
            "Web Image": "s"
        }
        function = function_mapping[self.function_var.get()]
        input_path = self.input_entry.get()
        output_path = self.output_entry.get()

        if not input_path or not output_path:
            messagebox.showerror("Error", "Please fill in both input and output paths.")
            return

        # Initialize the command list
        command = []

        if function == "i":
            command = ["rembg", "i", input_path, output_path]
        elif function == "p":
            command = ["rembg", "p", input_path, output_path]
        elif function == "s":
            # Handle web image with curl and rembg
            curl_command = f"curl -s {input_path}"
            rembg_command = "rembg i -om" if "-om" in [self.extra_params_options.get(self.extra_params_list.get(i)) for i in self.extra_params_list.curselection()] else "rembg i"
            full_command = f"{curl_command} | {rembg_command} > {output_path}"

            try:
                print("Running command:", full_command)
                subprocess.run(full_command, shell=True, check=True)
                self.result_label.config(text="Background removal completed successfully.")
                self.display_image(output_path)
            except subprocess.CalledProcessError as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

            return  # Skip further processing for web images

        # Append extra parameters
        selected_params = [self.extra_params_options[self.extra_params_list.get(i)] for i in self.extra_params_list.curselection()]
        command.extend(selected_params)

        try:
            print("Running command:", " ".join(command))
            subprocess.run(command, check=True)
            self.result_label.config(text="Background removal completed successfully.")
            self.display_image(output_path)
        except subprocess.CalledProcessError as e:
                        messagebox.showerror("Error", f"An error occurred: {e}")

    def open_output_folder(self):
        output_path = self.output_entry.get()
        folder = os.path.dirname(output_path)
        if platform.system() == "Windows":
            os.startfile(folder)
        elif platform.system() == "Darwin":
            subprocess.run(["open", folder])
        else:
            subprocess.run(["xdg-open", folder])

if __name__ == "__main__":
    app = RembgGUI()
    app.mainloop()
