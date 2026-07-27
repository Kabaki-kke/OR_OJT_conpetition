import  local_search
import bulk_dist_search
import first_step_search
import re_dijk

if __name__ == "__main__":
    work_dir = "case/japan"  # 例: "case/japan" や "kyuusyuu" など
    init_answer_folder = "local_minima_redijk"
    opt_answer_folder = "local_minima_redijk"
    # for i in range(10):
    #     print(f"計算 {i+1} セット目")
    #     max_iter = 20000    # 探索を試行する総回数
    #     improvements = local_search.main(work_dir, init_answer_folder, opt_answer_folder, max_iter)
    #     if improvements == 0:
    #         bulk_dist_search.main(work_dir, init_answer_folder, opt_answer_folder, 10000)
    for i in range(50):
        print(f"======= {i+1}/50 開始 =======")
        print("bulk_dist_search 開始")
        imp_bulk = bulk_dist_search.main(work_dir, init_answer_folder, opt_answer_folder, 5000)
        if imp_bulk == 0:
            print("first_step_search 開始")
            imp_first = first_step_search.main(work_dir, opt_answer_folder, opt_answer_folder,5000)        
            if imp_first == 0:
                print("local_search 開始")
                imp= local_search.main(work_dir, opt_answer_folder, opt_answer_folder, 20000)
                if imp == 0:
                    imp_redijk = re_dijk.main(work_dir, opt_answer_folder, opt_answer_folder, 1000)
                    if imp_redijk == 0:
                        break

        init_answer_folder = opt_answer_folder
                        
                
