import unittest
import train
import torch


def write_numbers_to_txt(numbers, filename):
    print("go2")
    # 打开文件以写入模式
    with open(filename, "w") as file:
        # 将数字列表转换为字符串，用空格分隔
        numbers_str = " ".join(map(str, numbers))
        # 写入字符串到文件
        print("go3")
        file.write(numbers_str)
    print("数字已写入到文件：", filename)


class TestReadFunc(unittest.TestCase):

    def test_get_train_and_val_ld(self):
        # print("go1")
        # write_numbers_to_txt([999, 2, 3, 4], "HW2/unit_test_file/ut1_id_and_label.txt")
        # tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).to(dtype=torch.int64)
        # torch.save(
        #     tensor1,
        #     "HW2/unit_test_file/999.pt",
        # )
        train_data_ld, validation_data_ld = train.get_train_and_val_ld(
            "HW2/unit_test_file/ut1_id_and_label.txt",
            "HW2/unit_test_file/",
            2,
            3,
        )
        train.train_model(train_data_ld, validation_data_ld, 2, 3)
        # for x, y in train_data_ld:
        #     print(f"train, x:{x},y:{y}")
        # for x, y in validation_data_ld:
        #     print(f"val, x:{x},y:{y}")


if __name__ == "__main__":
    unittest.main()
