#!/usr/bin/env python3
"""
unified_pressure_calculator_with_resistance.py
----------------------------------------------
Left panel: Resistance calculator using Poiseuille's Law
Right panel: Unified Pressure Calculator (Positive / Negative)
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.optimize import fsolve
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import time

# constants
Po_psi = 14.7
Po = Po_psi * 6894.76  # Pa
MAX_RUNTIME = 10.0  # seconds per curve


# ---------- Core Solvers ---------- #

def calc_curve_positive(Pm, t, R, Vo_ml):
    """Positive pressure (pushing) mode"""
    start = time.time()
    Vo = Vo_ml / 1e6
    Pi = Pm + 0.01
    Pt = 0.1

    def eqn(P, Pi, Vo, t, R):
        A = np.log(Pi - Po) - np.log(Pi) + (Po / Pi)
        return np.log(P - Po) - np.log(P) + Po / P - (Po / (Vo * R)) * t - A

    step = 0
    while Pt < Pm:
        if time.time() - start > MAX_RUNTIME:
            raise TimeoutError(f"Timeout for Vo={Vo_ml} mL (positive mode)")
        Pt_sol = fsolve(eqn, Pm, args=(Pi, Vo, t, R))
        Pt = Pt_sol[0]
        Pi += 0.01
        step += 1
        if step > 20000:
            raise RuntimeError(f"No convergence for Vo={Vo_ml} mL")

    # Generate pressure-time data
    A = np.log(Pt - Po) - np.log(Pt) + (Po / Pt)
    data = []
    P = Pt
    while P > (Pi - (2 * 6894.76)):
        t_graph = -(Vo * R / Po) * ((np.log(P - Po) - np.log(P) + (Po / P)) - A)
        data.append([t_graph, (P - Po) / 6894.76])
        P -= 0.01 * 6894.76

    df = pd.DataFrame(data, columns=["Time (s)", "Pressure (psi)"])
    if df["Time (s)"].iloc[-1] < t:
        lastP = df["Pressure (psi)"].iloc[-1]
        df = pd.concat([df, pd.DataFrame([[t, lastP]], columns=df.columns)], ignore_index=True)

    Vf = (Po / Pt) * Vo
    dV = (Vo - Vf) * 1e6  # mL
    Pt_g = (Pt / 6894.76) - 14.7
    return df, dV, Pt_g


def calc_curve_negative(Pm, t, R, Vo_ml):
    """Negative pressure (vacuum/suction) mode"""
    start = time.time()
    Vo = Vo_ml / 1e6
    Pi = Pm - 0.01
    Pt = Po - 100

    def eqn(P, Pi, Vo, t, R):
        A = np.log(Po - Pi) - np.log(Po) + (Pi / Po)
        return np.log(Po - P) - np.log(Po) + (P / Po) - (Po / (Vo * R)) * t - A

    step = 0
    while Pt > Pm:
        if time.time() - start > MAX_RUNTIME:
            raise TimeoutError(f"Timeout for Vo={Vo_ml} mL (negative mode)")
        Pt_sol = fsolve(eqn, Pm, args=(Pi, Vo, t, R))
        Pt = Pt_sol[0]
        Pi -= 0.01
        step += 1
        if step > 20000:
            raise RuntimeError(f"No convergence for Vo={Vo_ml} mL")

    A = np.log(Po - Pt) - np.log(Po) + (Pt / Po)
    data = []
    P = Pt
    while P < Po - 500:
        t_graph = -(Vo * R / Po) * ((np.log(Po - P) - np.log(Po) + (P / Po)) - A)
        data.append([t_graph, (P - Po) / 6894.76])
        P += 500

    df = pd.DataFrame(data, columns=["Time (s)", "Pressure (psi)"])
    if df["Time (s)"].iloc[-1] < t:
        lastP = df["Pressure (psi)"].iloc[-1]
        df = pd.concat([df, pd.DataFrame([[t, lastP]], columns=df.columns)], ignore_index=True)

    Vf = (Po / Pt) * Vo
    dV = abs(Vo - Vf) * 1e6
    Pt_g = (Pt / 6894.76) - 14.7
    return df, dV, Pt_g


# ---------- Resistance Calculator ---------- #

def calculate_resistance():
    try:
        length = float(entry_length.get())
        radius = float(entry_radius.get())
        viscosity = float(entry_viscosity.get())

        if radius <= 0 or length <= 0 or viscosity <= 0:
            raise ValueError("Inputs must be positive numbers.")

        R = (8 * viscosity * length) / (np.pi * (radius ** 4))
        result_res_label.config(text=f"R = {R:.3e} Pa·s/m²")

        # Auto-fill main calculator resistance
        entry_R.delete(0, tk.END)
        entry_R.insert(0, f"{R:.3e}")

    except Exception as e:
        messagebox.showerror("Error", f"Invalid resistance input:\n{e}")


# ---------- Main Calculator Logic ---------- #

def calculate_all():
    try:
        Pm_g = float(entry_Pm.get())
        t = float(entry_time.get())
        Vo_user = float(entry_volume.get())
        R = float(entry_R.get())
        mode = mode_var.get()

        Pm_abs = Po_psi + Pm_g
        Pm = Pm_abs * 6894.76

        # Remove old figure
        for w in right_frame.grid_slaves():
            if int(w.grid_info()["row"]) >= 7:
                w.destroy()

        vols = [Vo_user, 5, 10, 15, 20, 25]
        fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=100)
        axes = axes.flatten()

        for i, Vo_iter in enumerate(vols):
            ax = axes[i]
            try:
                if mode == "positive":
                    df, dV, Pt_g = calc_curve_positive(Pm, t, R, Vo_iter)
                    color_main, color_fill = "blue", "skyblue"
                else:
                    df, dV, Pt_g = calc_curve_negative(Pm, t, R, Vo_iter)
                    color_main, color_fill = "red", "salmon"

                ax.plot(df["Time (s)"], df["Pressure (psi)"], color=color_main, lw=1.2)
                fill_data = df[df["Time (s)"] <= t]
                ax.fill_between(fill_data["Time (s)"], fill_data["Pressure (psi)"],
                                color=color_fill, alpha=0.4)

                ax.set_xlim([0, t + 60])
                ax.set_xlabel("t (s)", fontsize=8)
                ax.set_ylabel("P (psi)", fontsize=8)
                ax.set_title(f"Vo={Vo_iter} mL | ΔV={dV:.3f} mL | Pi={Pt_g:.2f} psi", fontsize=9)
                ax.grid(True)

            except Exception as e:
                ax.text(0.5, 0.5, f"Failed\n{e}", ha="center", va="center",
                        fontsize=7, color="red")
                ax.axis("off")

        plt.tight_layout(pad=2.0)

        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=7, column=0, columnspan=2)

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input:\n{e}")


# ---------- GUI Setup ---------- #

root = tk.Tk()
root.title("Unified Pressure Calculator + Resistance Calculator")

# Split into two main frames
left_frame = tk.Frame(root, padx=10, pady=10, relief="ridge", bd=3)
left_frame.grid(row=0, column=0, sticky="nsew")
right_frame = tk.Frame(root, padx=10, pady=10, relief="ridge", bd=3)
right_frame.grid(row=0, column=1, sticky="nsew")

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)

# --- Left: Resistance Calculator ---
tk.Label(left_frame, text="Resistance Calculation", font=("Times New Roman", 18, "bold")).grid(row=0, column=0, columnspan=2, pady=10)

tk.Label(left_frame, text="Length [m]:", font=("Times New Roman", 14)).grid(row=1, column=0, sticky="e", pady=5)
entry_length = tk.Entry(left_frame, font=("Times New Roman", 14), width=10)
entry_length.grid(row=1, column=1, pady=5)

tk.Label(left_frame, text="Radius [m]:", font=("Times New Roman", 14)).grid(row=2, column=0, sticky="e", pady=5)
entry_radius = tk.Entry(left_frame, font=("Times New Roman", 14), width=10)
entry_radius.grid(row=2, column=1, pady=5)

tk.Label(left_frame, text="Viscosity [Pa·s]:", font=("Times New Roman", 14)).grid(row=3, column=0, sticky="e", pady=5)
entry_viscosity = tk.Entry(left_frame, font=("Times New Roman", 14), width=10)
entry_viscosity.grid(row=3, column=1, pady=5)

tk.Button(left_frame, text="Calculate Resistance", command=calculate_resistance,
          font=("Times New Roman", 14, "bold"), bg="#4682B4", fg="black").grid(row=4, column=0, columnspan=2, pady=15)

result_res_label = tk.Label(left_frame, text="R = ", font=("Times New Roman", 14, "bold"), fg="white")
result_res_label.grid(row=5, column=0, columnspan=2, pady=5)

# --- Right: Main Pressure Calculator ---
font_label = ("Times New Roman", 16)
font_entry = ("Times New Roman", 16)
font_button = ("Times New Roman", 18, "bold")

mode_var = tk.StringVar(value="positive")
frame_mode = tk.Frame(right_frame)
frame_mode.grid(row=0, column=0, columnspan=2, pady=5)
tk.Label(frame_mode, text="Mode:", font=("Times New Roman", 16, "bold")).pack(side="left", padx=5)
tk.Radiobutton(frame_mode, text="Positive Pressure", variable=mode_var, value="positive",
               font=("Times New Roman", 14)).pack(side="left", padx=10)
tk.Radiobutton(frame_mode, text="Negative Pressure", variable=mode_var, value="negative",
               font=("Times New Roman", 14)).pack(side="left", padx=10)

# Inputs
tk.Label(right_frame, text="Target Pressure [psi gauge]:", font=font_label).grid(row=1, column=0, sticky="e", padx=5, pady=5)
entry_Pm = tk.Entry(right_frame, font=font_entry, width=10); entry_Pm.grid(row=1, column=1, pady=5)

tk.Label(right_frame, text="Time [s]:", font=font_label).grid(row=2, column=0, sticky="e", padx=5, pady=5)
entry_time = tk.Entry(right_frame, font=font_entry, width=10); entry_time.grid(row=2, column=1, pady=5)

tk.Label(right_frame, text="User Initial Volume [mL]:", font=font_label).grid(row=3, column=0, sticky="e", padx=5, pady=5)
entry_volume = tk.Entry(right_frame, font=font_entry, width=10); entry_volume.grid(row=3, column=1, pady=5)

tk.Label(right_frame, text="Resistance [Pa·s/m²]:", font=font_label).grid(row=4, column=0, sticky="e", padx=5, pady=5)
entry_R = tk.Entry(right_frame, font=font_entry, width=10); entry_R.grid(row=4, column=1, pady=5)

tk.Button(right_frame, text="Generate Graph", command=calculate_all,
          font=font_button, bg="#2E8B57", fg="black", width=12, height=1).grid(row=5, column=0, columnspan=2, pady=15)

tk.Label(right_frame, text="", font=font_label).grid(row=6, column=0, columnspan=2)  # spacer

root.mainloop()
