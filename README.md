# LANL-Earthquake-Prediction

## Github协作

+ 不得直接修改主分支`master`
+ 个人分支命名为`dev_xxx`
+ 协作流程
  + 创建新分支`git branch dev_xxx`
  + 查看所有分支`git branch`
  + 切换分支`git checkout dev_xxx`（确保本地仓库为个人分支的内容）
  + 拉取仓库`git pull origin master`（从远端仓库的主分支拉取内容到本地仓库的个人分支）
  + 首次推送仓库`git push --set-upstream origin dev_xxx `（将本地仓库的个人分支绑定到远端仓库的个人分支）
  + 再次推送仓库`git push`（将本地仓库的个人分支推送到远端仓库的个人分支）
  + 在Github上通过`pull request`将个人分支的修改更新到主分支