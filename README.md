# 杨少侠的写字板

一个使用 Jekyll 构建、适合托管到 GitHub Pages 的个人博客。公开文章目录目前为空，首页会显示留白状态。

## 本地预览

需要 Ruby 和 Bundler：

```bash
bundle install
bundle exec jekyll serve --drafts --livereload
```

打开 `http://127.0.0.1:4000`。`--drafts` 会同时显示 `_drafts` 里的草稿；正式构建不会发布草稿。

## 写一篇文章

1. 在 `_posts/` 中新建 `YYYY-MM-DD-article-slug.md`，例如 `_posts/2026-11-03-position-encoding-to-roformer.md`。
2. 把下面的头部信息放在文件开头，再使用 Markdown 写正文：

```markdown
---
title: "文章标题"
description: "一两句话概括文章，会显示在列表和搜索结果中。"
tags: [机器学习, Transformer]
category: 深度学习
icon: wave
math: true
---

正文从这里开始。
```

`category` 可设为 `深度学习`、`AGENTS` 或 `复杂系统`，用于首页栏目筛选。不需要公式时删除 `math: true`。使用 `$...$` 写行内公式，使用 `$$...$$` 写独立公式。`icon` 可选 `wave`、`nodes`、`orbit`、`steps`、`sun`，不写时自动轮换。

已经准备了第一篇草稿：`_drafts/position-encoding-to-roformer.md`。写完后，把它移动到 `_posts/`，并在文件名前加上发布日期即可公开。

提交并推送后，GitHub Pages 会自动更新：

```bash
git add .
git commit -m "Add position encoding notes"
git push
```

## 修改个人信息

主要信息都在 `_config.yml`：

- `title`：站点名称
- `description`：站点简介
- `url`：线上地址
- `author.name`：作者名称
- `github_username`：GitHub 用户名

首页文案在 `index.html`，个人介绍在 `about.html`，颜色与字体变量位于 `assets/css/main.css` 顶部。

## 发布到 GitHub Pages

GitHub 仓库名必须是 `yangzc2004-bit.github.io`，发布后的网址为 `https://yangzc2004-bit.github.io/`。

当前电脑已经通过 GitHub CLI 登录。首次发布可执行：

```bash
git add .
git commit -m "Build my personal blog"
gh repo create yangzc2004-bit.github.io --public --source=. --remote=origin --push
```

然后在仓库的 **Settings → Pages** 中选择 **Deploy from a branch**，分支选择 `main`，目录选择 `/(root)`。保存后等待部署完成。

> 创建公开仓库和推送属于线上操作，命令由你确认后再执行。

## 功能

- 响应式首页、文章归档、关于页和 404 页面
- Markdown 文章、代码高亮、MathJax 数学公式
- 文章页自动目录、上一篇与下一篇导航
- 全站文章搜索，快捷键 `⌘ K` / `Ctrl K`
- 明暗主题切换、RSS、SEO 与站点地图
