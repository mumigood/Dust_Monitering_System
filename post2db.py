import numpy as np
import cv2
import pandas as pd
import random
import pickle
from datetime import datetime
import time
import sqlite3
import os
import matplotlib.pyplot as plt
import subprocess
import yaml


class UserCaseException(Exception):
    pass


class Dust_Monitor(object):
    def __init__(self, rtsp_site=None):
        self.config_dict = self.process_config_file()

        if self.config_dict["to_db"].lower() == "true":
            self.to_db = True
        else:
            self.to_db = False

        if self.config_dict["rtsp_url"] is not None:
            self.cap = cv2.VideoCapture(self.config_dict["rtsp_url"])
        else:
            self.cap = cv2.VideoCapture(rtsp_site)

        if (
            os.path.isfile("./light/Red/OFF/Debug/Comport.exe") == False
            or os.path.isfile("./light/Red/ON/Debug/Comport.exe") == False
        ):
            raise UserCaseException("Red light執行檔不存在!!")
        self.light_exe = {
            "red": {
                "on": os.path.join("./light/Red/ON/Debug/Comport.exe"),
                "off": os.path.join("./light/Red/OFF/Debug/Comport.exe"),
            }
        }

        if "params.pkl" not in os.listdir("."):
            raise UserCaseException("params.pkl不存在!!")
        with open("./params.pkl", "rb") as f:
            params = pickle.load(f)
            self.contrast_var_list = params["contrast_var_list"]
            self.init_hist = params["hist"]

        contrast_var_mv = self.moving_average(self.contrast_var_list, 100)
        self.max_min = {
            "contrast": {"max": max(contrast_var_mv), "min": min(contrast_var_mv)},
            "hist": {"max": 1.0, "min": 0.0},
        }

        if "dust_data.db" not in os.listdir("."):
            raise UserCaseException("dust_data.db不存在!!")
        self.db_file, self.table_name, self.db_file_simulation = (
            "dust_data.db",
            "dust_data",
            "dust_data_simulation.db",
        )

        self.val_list, self.idx = [], random.randint(0, 6177 - 80)

    def process_config_file(self):
        if "config.yaml" not in os.listdir("."):
            raise UserCaseException("config.yaml不存在!!")
        with open("config.yaml", "r", encoding="utf-8") as f:
            lines = f.readlines()

        filtered_lines = []
        skip_block = False
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("settings:"):
                skip_block = True
            elif skip_block and not stripped_line:
                skip_block = False
            elif not skip_block:
                filtered_lines.append(line)

        filtered_text = "".join(filtered_lines)
        config_dict = yaml.safe_load(filtered_text)

        if "thresholds" in config_dict:
            config_dict["thresholds"] = {
                key.replace("_line", ""): value
                for key, value in config_dict["thresholds"].items()
            }
        return config_dict

    def data2db(self, val, mode="simulation"):
        current_time = datetime.now()
        df = pd.DataFrame([current_time.strftime("%Y-%m-%d %H:%M:%S"), val]).T
        df.columns = ["Timestamp", "Dust_Level"]

        if mode == "simulation":
            conn = sqlite3.connect(self.db_file_simulation)
        elif mode == "shot":
            conn = sqlite3.connect(self.db_file)
        else:
            raise UserCaseException("請輸入shot或simulation!!")
        df.to_sql(self.table_name, conn, if_exists="append", index=False)
        conn.close()

    def moving_average(self, x, w):
        return np.convolve(x, np.ones(w), "valid") / w

    def test2db(self, w=70):
        conn = sqlite3.connect(self.db_file_simulation)
        original_df = pd.read_sql_query(
            "SELECT * FROM {};".format(self.table_name), conn
        )

        for _ in range(0, w * 100):
            self.val_list.append(self.contrast_var_list[self.idx])
            self.idx += 1

            if len(self.val_list) == w:
                val_mv = self.moving_average(self.val_list, w)
                normalized_val_mv = self.normalization(val_mv)
                self.data2db(normalized_val_mv)
                self.val_list, self.idx = [], random.randint(0, 6177 - 80)

        conn = sqlite3.connect(self.db_file_simulation)
        modify_df = pd.read_sql_query("SELECT * FROM {};".format(self.table_name), conn)

        if len(modify_df) <= len(original_df):
            raise UserCaseException("模擬資料沒有存進db!!")
        else:
            print("模擬資料有存進db!!")

    def normalization(self, val, algorithm="contrast"):
        scale, bias = (
            100 / abs(self.max_min[algorithm]["max"] - self.max_min[algorithm]["min"]),
            self.max_min[algorithm]["max"],
        )
        normalized_val = scale * (-val + bias)
        if normalized_val > 100:
            return 100
        elif normalized_val < 0:
            return 0
        return normalized_val

    def algorithm_contrast(self, image):
        dust_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dust_gray = cv2.convertScaleAbs(dust_gray, alpha=1, beta=50)
        laplacian = cv2.Laplacian(dust_gray, cv2.CV_64F)
        variance_of_laplacian = laplacian.var()
        return variance_of_laplacian

    def algorithm_hist(self, image):
        dust_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([dust_gray], [0], None, [256], [0, 256])
        hist /= hist.sum()
        correlation = cv2.compareHist(hist, self.init_hist, cv2.HISTCMP_CORREL)
        return correlation

    def record_init_hist(self, fps=7, sec=20):
        self.init_hist = []
        while self.cap.isOpened() == True and len(self.init_hist) < fps * sec:
            status, frame = self.cap.read()
            image = cv2.resize(frame, (1000, 750), interpolation=cv2.INTER_AREA)
            image_crop = image[230:300, 310:410]
            dust_gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([dust_gray], [0], None, [256], [0, 256])
            hist /= hist.sum()
            self.init_hist.append(hist.reshape(-1))

        self.init_hist = np.array(self.init_hist)
        self.init_hist = self.init_hist.mean(axis=0).reshape([-1, 1])

    def vedio_stream(self, w=70, cv2_show=True):
        while self.cap.isOpened() == True:
            status, frame = self.cap.read()
            image = cv2.resize(frame, (1000, 750), interpolation=cv2.INTER_AREA)
            image_crop = image[230:300, 310:410]
            val = self.algorithm_hist(image_crop)  # 演算法部分

            # 1.self.val_list的長度等於w時
            # 2.將self.val_list做移動平均
            # 3.將做移動平均的結果存入db
            # 4.清空self.val_list
            self.val_list.append(val)
            if len(self.val_list) >= w:
                val_mv = self.moving_average(self.val_list, w)[0]
                normalized_val_mv = self.normalization(val_mv)
                if self.to_db == True:
                    print("存入db")
                    self.(normalized_val_mv, mode="shot")
                self.val_list = []

            if cv2_show == True:
                cv2.imshow("Webcam", image)
                cv2.namedWindow("crop", 0)
                cv2.resizeWindow("crop", 400, 280)
                cv2.imshow("crop", image_crop)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("退出程式")
                    break
                elif key == ord("s"):
                    current_time = datetime.now()
                    cv2.imwrite(
                        "./save_image/{}.jpg".format(
                            current_time.strftime("%Y-%m-%d %H-%M-%S")
                        ),
                        frame,
                    )
                    print(
                        "{}.jpg 存檔".format(current_time.strftime("%Y-%m-%d %H-%M-%S"))
                    )

        self.cap.release()
        cv2.destroyAllWindows()


tt = Dust_Monitor(rtsp_site="rtsp://localhost:8554/mystream")
# tt.test2db()
tt.vedio_stream()
