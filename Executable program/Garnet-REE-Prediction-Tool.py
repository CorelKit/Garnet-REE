import pickle
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.ensemble import RandomForestClassifier

def detect_anomalies(row, columns_order):
    y_values = row[columns_order].values.astype(float)
    n = len(y_values)
    #  6 La, 7 Ce, 8 Pr, 9 Nd*, 10 Sm,
    #  11 Eu, 12 Gd, 13 Tb*, 14 Dy, 5 Y,
    #  15 Ho, 16 Er*, 17 Tm*, 18 Yb*, 19 Lu,
    lower = [1, 0.0001, 0.0001, 0.4, 0.0001,
             0.0001, 0.0001, 0.4, 0.0001, 0.0001,
             0.0001, 0.4, 0.4, 0.4, 1]
    upper = [1, 10000, 10000, 1.6, 10000,
             10000, 10000, 1.6, 10000, 10000,
             10000, 1.6, 1.6, 1.6, 1]
    for i in range(n):
        if i == 0:  # 首点用后邻值
            geo_mean = y_values[i]  # y_values[i + 1]
        elif i == n - 1:  # 末点用前邻值
            geo_mean = y_values[i]  # y_values[i - 1]
        else:  # 中间点用前后几何平均
            geo_mean = (y_values[i - 1] * y_values[i + 1]) ** 0.5
        ratio = y_values[i] / geo_mean
        if not (lower[i] <= ratio <= upper[i]):
            return True
    return False

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

class ExcelProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("A Prediction Tool for Skarn Metallic Mineralization Types Based on Garnet Rare Earth Elements")
        self.root.geometry("800x600")

        # Store data
        self.df = None
        self.processed_df = None

        # Create an interface
        self.create_widgets()

    def create_widgets(self):
        # Button frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10, fill=tk.X)

        # Button
        ttk.Button(button_frame, text="Import xlsx file", command=self.load_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Process data and predict", command=self.process_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save result", command=self.save_results).pack(side=tk.LEFT, padx=5)

        # Separator line
        ttk.Separator(self.root).pack(fill=tk.X, pady=5)

        # Result display framework
        result_frame = ttk.Frame(self.root)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Notebook (Tab)
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ===================== Raw data tab =====================
        self.raw_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.raw_frame, text="Raw Data")

        # Container framework
        raw_container = ttk.Frame(self.raw_frame)
        raw_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Raw data table
        self.raw_tree = ttk.Treeview(raw_container)
        self.raw_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a vertical scroll bar
        raw_vsb = ttk.Scrollbar(raw_container, orient="vertical", command=self.raw_tree.yview)
        raw_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.raw_tree.configure(yscrollcommand=raw_vsb.set)

        # Add a horizontal scroll bar
        raw_hsb = ttk.Scrollbar(self.raw_frame, orient="horizontal", command=self.raw_tree.xview)
        raw_hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.raw_tree.configure(xscrollcommand=raw_hsb.set)

        # ===================== Processing result tab =====================
        self.processed_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.processed_frame, text="Processed Results")

        # Container framework
        processed_container = ttk.Frame(self.processed_frame)
        processed_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Processing result table
        self.processed_tree = ttk.Treeview(processed_container)
        self.processed_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a vertical scroll bar.
        processed_vsb = ttk.Scrollbar(processed_container, orient="vertical", command=self.processed_tree.yview)
        processed_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.processed_tree.configure(yscrollcommand=processed_vsb.set)

        # Add a horizontal scroll bar
        processed_hsb = ttk.Scrollbar(self.processed_frame, orient="horizontal", command=self.processed_tree.xview)
        processed_hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.processed_tree.configure(xscrollcommand=processed_hsb.set)

        # ===================== Log tab =====================
        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text="Operation Log")

        # Container framework
        log_container = ttk.Frame(self.log_frame)
        log_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Log text box
        self.log_text = tk.Text(log_container, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a vertical scroll bar.
        log_vsb = ttk.Scrollbar(log_container, orient="vertical", command=self.log_text.yview)
        log_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_vsb.set, state=tk.DISABLED)

        # ===================== Status bar =====================
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.update_status("Ready")

        # Set the default displayed tab to "Operation Log".
        self.notebook.select(self.log_frame)

    def update_status(self, message):
        self.status_var.set(message)
        self.log_message(f"Status Update: {message}")

    def log_message(self, message):
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)  # 自动滚动到底部

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Excel", "*.xlsx"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            self.df = pd.read_excel(file_path)
            self.update_status(f"File loaded successfully: {os.path.basename(file_path)}")
            self.log_message(f"File imported: {file_path}")
            self.log_message(f"Data dimensions:: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            self.display_data(self.raw_tree, self.df)
            REEs = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y']
            columns_to_check = self.df.columns[2:17]
            is_match = list(columns_to_check) == REEs
            if is_match:
                self.log_message("The column headers from column 3 to 17 match the Rare Earth Elements list.")
            else:
                self.log_message("The column headers from column 3 to 17 do not match the Rare Earth Elements list.")
                self.log_message(f"Expected: {REEs}")
                self.log_message(f"Actual: {list(columns_to_check)}")
        except Exception as e:
            messagebox.showerror("Loading Error", f"Unable to load file:\n{str(e)}")
            self.log_message(f"Loading Error: {str(e)}")

    def display_data(self, treeview, dataframe):
        # Clear the existing data.
        treeview.delete(*treeview.get_children())
        columns = list(dataframe.columns)
        treeview["columns"] = columns
        treeview["show"] = "headings"
        for col in columns:
            treeview.heading(col, text=col)
            treeview.column(col, width=100, anchor=tk.CENTER)
        for _, row in dataframe.iterrows():
            treeview.insert("", tk.END, values=list(row))

    def process_data(self):
        if self.df is None or self.df.empty:
            messagebox.showwarning("No Data", "Please import the Excel file first")
            return

        try:
            self.log_message("Starting to process the data...")
            sel_columns = self.df.copy()
            numeric_cols_positions = list(range(2, 17))
            numeric_columns = [sel_columns.columns[pos] for pos in numeric_cols_positions]
            sel_columns = sel_columns.dropna(subset=numeric_columns)
            rows_rawnumber = sel_columns.shape[0]

            # delete BDL data
            if (sel_columns[numeric_columns] <= 0).any().any():
                sel_columns[numeric_columns] = sel_columns[numeric_columns].apply(
                    lambda x: np.where(x > 0, x, np.nan))
                sel_columns = sel_columns.dropna(subset=numeric_columns)
            rows_removeBDL = sel_columns.shape[0]
            self.log_message(f"Deleted data below the detection limit: {rows_rawnumber - rows_removeBDL}, Remaining data: {rows_removeBDL}.")

            # REE normalized
            Chondrite_S_M_1989 = [0.237, 0.612, 0.095, 0.467, 0.153,  # La Ce Pr Nd Sm
                                  0.058, 0.2055, 0.0374, 0.254, 0.0566,  # Eu Gd Tb Dy Ho
                                  0.1655, 0.0255, 0.17, 0.0254, 1.57, ]  # Er Tm Yb Lu Y
            columns_to_process = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y']
            chondrite_series = pd.Series(Chondrite_S_M_1989, index=columns_to_process)
            sel_columns[columns_to_process] = sel_columns[columns_to_process].div(chondrite_series)
            self.log_message("The rare earth elements are normalized to chondritic values (Sun and McDonough 1989).")

            # Deleted abnormal REEs pattern data
            columns_order2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 11, 12, 13, 14, 15]
            mask_clean = sel_columns.apply(lambda row: not detect_anomalies(row, columns_order2), axis=1)
            sel_columns_clean = sel_columns[mask_clean]
            sel_columns_clean = sel_columns_clean.reset_index(drop=True)
            rows_clean = sel_columns_clean.shape[0]
            self.log_message(f"Deleted abnormal rare earth element pattern data: {rows_removeBDL - rows_clean}; remaining data: {rows_clean}.")

            # PCA
            current_dir = get_base_path()
            with open(current_dir+ '/models/PCA.pkl', 'rb') as pkl_path:
                pca_out = pickle.load(pkl_path)
            pca_scores = pca_out.transform(np.log10(sel_columns_clean.iloc[:, 2:17]))
            pc_list = ["PC" + str(i) for i in list(range(1, 7))]
            PCA_df = pd.DataFrame(pca_scores, columns=pc_list)
            self.log_message("PCA dimensionality reduction has been completed, yielding six principal components (≥99% cumulative variance explained).")

            # Prediction Random Forest
            metal = ['W', 'Sn', 'Mo', 'Fe', 'Cu', 'Au', 'Pb', 'Zn']
            blind_v, countnum = [], 0
            for mymetal in metal:
                countnum += 1
                pkl_path = current_dir + '/models/Random Forest_SM_' + mymetal + '.pkl'

                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        loaded_model = pickle.load(f)
                else:
                    print(pkl_path, ' does not exist')
                    exit()
                y_val = loaded_model.predict(PCA_df)
                if countnum == 1:
                    blind_v = y_val.reshape(-1, 1)  # Initialize as a two-dimensional array
                else:
                    blind_v = np.hstack((blind_v, y_val.reshape(-1, 1)))  # Horizontal stacking
            metal_with_rf = [f"{m}(rf)" for m in metal]
            results_rf_df = pd.DataFrame(blind_v, columns=metal_with_rf)
            results_rf_df.replace({0: "B-W", 1: "S"}, inplace=True)
            count_stats = []
            for col in results_rf_df.columns:
                count_0 = (blind_v[:, results_rf_df.columns.get_loc(col)] == 0).sum()
                count_1 = (blind_v[:, results_rf_df.columns.get_loc(col)] == 1).sum()
                count_stats.append(f"{col}: B-W={count_0}, S={count_1}")
            self.log_message("Random Forest (rf) is used to build the model.")
            self.log_message(f"Predicted result matrix: "
                             f"\nB-W: Barren to Weak Metallic Mineralization; "
                             f"\nS: Strong Metallic Mineralization."
                             f"\nMetal thresholds (T). "
                             f"\nW 5 kt, Sn 1 kt, Mo 10 kt, Fe 4 Mt,"
                             f"\nCu 0.2 Mt, Au 10 t, Pb 0.1 kt, Zn 0.1 kt."
                             f"\nB-W < T, S >= T."
                             f"\n{str(results_rf_df)}")
            self.log_message("Statistical results:\n"+"\n".join(count_stats))

            # Display the processed data.
            df_zcp = pd.concat(
                [sel_columns_clean.reset_index(drop=True),
                 PCA_df.iloc[:, 0:7].reset_index(drop=True),
                 # results_xgb_df.reset_index(drop=True),
                 results_rf_df.reset_index(drop=True),
                 ],
                axis=1)
            self.processed_df = df_zcp

            self.display_data(self.processed_tree, self.processed_df)

            self.log_message("Data processing completed")
            self.update_status("Data processing completed")

        except Exception as e:
            messagebox.showerror("Processing Error", f"Data processing failed:\n{str(e)}")
            self.log_message(f"Processing Error: {str(e)}")

    def save_results(self):
        if self.processed_df is None or self.processed_df.empty:
            messagebox.showwarning("No Data", "There is no processed data to save")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            self.processed_df.to_excel(file_path, index=False)
            self.log_message(f"The result has been saved to: {file_path}")
            self.update_status(f"File saved: {os.path.basename(file_path)}")
            messagebox.showinfo("Save Successful", "The result has been successfully saved as an Excel file")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save file:\n{str(e)}")
            self.log_message(f"Save Error: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ExcelProcessorApp(root)
    root.mainloop()
