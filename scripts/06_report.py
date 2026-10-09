"""Generate Chinese Markdown and a paginated PDF from the saved REAL results."""
import html
import json
import re
import os
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
from common import ROOT, DATA, RESULTS, FIGURES, CLASSES, GROUPS, KEYWORDS, read_json

stats=read_json(RESULTS/'corpus_stats.json')
emb=read_json(RESULTS/'embedding_analysis.json')
training=read_json(RESULTS/'embedding_training.json')
runs=read_json(RESULTS/'classification_runs.json')
summary=read_json(RESULTS/'classification_summary.json')
preds=read_json(RESULTS/'predictions.json')
coverage=read_json(RESULTS/'pretrained_coverage.json')
paired=read_json(RESULTS/'paired_comparison.json')
verification=read_json(RESULTS/'verification.json')
articles={a['id']:a for a in (json.loads(x) for x in (DATA/'articles.jsonl').read_text().splitlines())}
def run(k,method,seed=42):
    return next(r for r in runs if r['classes']==k and r['method']==method and r['seed']==seed and r['train_per_class']==48 and not r['repeat'])
def summ(k,method,n=48):
    return next(s for s in summary if s['classes']==k and s['method']==method and s['train_per_class']==n)
def score(r): return f"{r['accuracy']:.3f} / {r['macro_f1']:.3f}"
def pm(s,metric): return f"{s[metric+'_mean']:.3f} ± {s[metric+'_std']:.3f}"
custom_font=os.environ.get('HW2_PDF_FONT')
songti=Path('/System/Library/Fonts/Supplemental/Songti.ttc')
if custom_font:
    pdfmetrics.registerFont(TTFont('CJK',custom_font))
    FONT='CJK'
elif songti.exists():
    pdfmetrics.registerFont(TTFont('CJK',str(songti),subfontIndex=6))
    FONT='CJK'
else:
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    FONT='STSong-Light'
BODY=ParagraphStyle('Body',fontName=FONT,fontSize=10.3,leading=16.5,textColor=colors.HexColor('#243447'),spaceAfter=9,wordWrap='CJK')
HEAD=ParagraphStyle('Head',parent=BODY,fontSize=18,leading=25,spaceAfter=15,textColor=colors.HexColor('#125C69'))
SUB=ParagraphStyle('Sub',parent=BODY,fontSize=12.5,leading=19,spaceBefore=7,spaceAfter=8,textColor=colors.HexColor('#125C69'))
CELL=ParagraphStyle('Cell',parent=BODY,fontSize=8.6,leading=12,spaceAfter=0)
SMALL=ParagraphStyle('Small',parent=BODY,fontSize=8.5,leading=13,spaceAfter=7)
CODE=ParagraphStyle('Code',parent=SMALL,fontName='Courier',fontSize=7.2,leading=11,wordWrap='CJK',backColor=colors.HexColor('#F1F5F7'),borderPadding=8)
story=[]; md=[]
def esc(s): return html.escape(str(s))
def p(text,small=False):
    story.append(Paragraph(esc(text),SMALL if small else BODY));md.append(text+'\n')
def h(text):
    story.append(Paragraph(esc(text),SUB));md.append('## '+text+'\n')
def page(title,first=False):
    if not first: story.append(PageBreak());md.append('\n---\n')
    story.append(Paragraph(esc(title),HEAD));md.append('# '+title+'\n')
def table(headers,rows,widths=None):
    data=[[Paragraph(esc(v),CELL) for v in headers]]+[[Paragraph(esc(v),CELL) for v in row] for row in rows]
    tab=Table(data,colWidths=widths or [480/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
    tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E6F1F2')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F6F8FA')]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#A8C9CD'))]))
    story.extend([tab,Spacer(1,10)])
    md.append('| '+' | '.join(map(str,headers))+' |\n| '+' | '.join(['---']*len(headers))+' |')
    md.extend('| '+' | '.join(str(x).replace('|','／').replace('\n',' ') for x in row)+' |' for row in rows);md.append('')
def image(name,caption,width=480):
    from reportlab.lib.utils import ImageReader
    path=FIGURES/name; iw,ih=ImageReader(str(path)).getSize()
    story.append(Image(str(path),width=width,height=width*ih/iw));p(caption,small=True)
    md.append(f'![{caption}](figures/{name})\n')
def code(text):
    story.append(Paragraph('<br/>'.join(esc(s).replace(' ','&#160;') for s in text.splitlines()),CODE));story.append(Spacer(1,10));md.append('```bash\n'+text+'\n```\n')

page('Gensim 词表示学习与 fastText 文本分类',True)
p('编程作业实验报告 | 实测日期：2026-10-07 | 所有数值由本次运行的 JSON 和逐篇预测生成。',small=True)
repository_url='https://github.com/JennyMa11/hw2-gensim-fasttext'
story.append(Paragraph(f'GitHub 仓库：<link href="{repository_url}" color="#176B75">{repository_url}</link>',BODY))
md.append(f'GitHub 仓库：[{repository_url}]({repository_url})\n')
h('1. 实验目标与方法')
p('本实验比较 Word2Vec 与 FastText 的中文词表示，并检验预训练向量对小样本文本分类的作用。基础部分完成近邻、词表外查询、词组余弦分析，以及嵌套的 2／4／8 类监督分类；扩展部分比较 Word2Vec 初始化，并绘制训练样本数的学习曲线。')
p('Word2Vec 为词表内的每个词学习向量；skip-gram 根据中心词预测周围词。本次使用负采样，以真实上下文为正例、按词频的 0.75 次幂采样的词为负例。窗口上限为 5，有效窗口随机缩小，高频词下采样阈值为 0.001。')
p('FastText 在词向量之外学习字符 n-gram 向量，同一字符片段会参与多个词的表示。这样能为未进入词表的词合成向量，也可能过度强调字面相似。近邻与余弦只反映语料统计结构，不能直接等同于词义正确性。')
p('Gensim FastText 负责无监督词向量学习；分类使用官方 fastText 的 train_supervised。导出的 .vec 仅含已合成的词表内向量，没有迁移 Gensim 的字符桶参数。分类器再用带标签正文训练 softmax，所有导入向量都可以继续更新。')
h('共同表示学习参数')
table(['设置','取值'],[
    ['算法／维度','skip-gram／100'],['窗口／最低频次／轮次','5／3／5'],
    ['负采样／下采样','5 个负例；ns_exponent=0.75；sample=0.001'],
    ['可复现设置','seed=42；workers=1；PYTHONHASHSEED=0'],
    ['FastText 特有设置','字符 n-gram 长度 2-4；bucket=200000']], [125,355])
p('实现依据：Mikolov 等（2013）、Bojanowski 等（2017）、Joulin 等（2017）；文末列出论文和官方文档。实验独立编写并实际执行，没有沿用参考作业中的预测或分数。',small=True)

page('2. 语料、标签与防泄漏划分')
p('采用 PeopleDaily1998 公开镜像中的 199801.zip。包内实际包含 1998 年 1-6 月，不按文件名推断月份。与题目示例的 THUCNews 设置不同，本实验用人民日报完成两个阶段，并在词向量训练前排除全部分类测试文章。')
table(['处理项目','本次统计'],[
    ['恢复文章数',f"{stats['raw_articles']:,}"],['正文不足 30 词／重复正文',f"{stats['parsing']['short_body_articles']:,}／{stats['parsing']['duplicate_body_articles']}"],
    ['保留的非重复文章',f"{stats['eligible_unique_articles']:,}"],['预训练排除／实际使用文章',f"96／{stats['representation_articles']:,}"],
    ['表示学习句子／词次',f"{stats['representation_sentences']:,}／{stats['representation_tokens']:,}"],
    ['原始词表／min_count=3 后',f"{stats['raw_vocabulary']:,}／{stats['vocabulary_min_count_3']:,}"]],[190,290])
p('以编号的日期、版面、文章序号恢复文章；001 行作为标题，其后作为正文。去掉编号、词性、标点和实体边界，保留复合实体内部现成的分词，不再用 jieba。按句末标点分句，舍弃不足 3 词的片段。1,334 个多斜杠词元按首个词性分隔符处理；唯一空词 /m 被丢弃。')
p('语料没有主题金标签。使用公开在 common.py 的标题关键词生成弱标签：只接受命中一个主题、标题长 4-80 字且正文不少于 80 词的文章。跨主题候选共 164 篇被排除。未对全部样本做人工金标准标注。分类输入去掉标题，仅保留正文前 500 词。')
table(['类别','可用候选','训练／测试'],[[c,stats['candidates'][c],'48／12'] for c in CLASSES],[150,140,190])
p('每类按 SHA256(sample42:文章ID) 排序取前 60 篇，再按 SHA256(split42:文章ID) 排序取前 12 篇为测试。2 类为体育、金融；4 类再加教育、文化；8 类再加科技、医疗、农业、法治。三套实验沿用相同文章和划分，样本 ID、标题、命中词与正文哈希均在 sample_manifest.csv。',small=True)

page('3. 近邻与词表外查询')
p(f"两个模型最终词表均为 {training['word2vec']['vocabulary']:,} 词。Apple M5、16 GB 内存环境中，Word2Vec 本次构建词表并训练耗时 {training['word2vec']['seconds']:.1f} 秒，FastText 为 {training['fasttext']['seconds']:.1f} 秒；不含保存、导出与首次导入库的时间。单机单次耗时不代表通用速度排名。")
def neigh(items,n=3): return '；'.join(f'{w} {s:.3f}' for w,s in items[:n])
table(['词（语料频次）','Word2Vec 前三近邻','FastText 前三近邻'],[[f"{w}（{emb['word2vec']['queries'][w]['frequency']}）",neigh(emb['word2vec']['queries'][w]['neighbors']),neigh(emb['fasttext']['queries'][w]['neighbors'])] for w in emb['word2vec']['queries']],[95,192.5,192.5])
p('近邻既受上下文影响，也受分词、频次和子词共享影响。FastText 更容易产生共享连续汉字的邻居；这是一种表示机制，不足以证明语义总体更好。完整 top-5 近邻及余弦数值保存在 embedding_analysis.json。')
h('词表外（OOV）查询')
table(['查询词','训练频次／词表内','Word2Vec','FastText 前三近邻'],[[w,f"{v['frequency']}／{v['in_vocabulary']}",emb['word2vec']['oov'][w].get('error','可查询'),neigh(v['neighbors'])] for w,v in emb['fasttext']['oov'].items()],[95,80,65,240])
p('以上构造词的词表状态通过 key_to_index 判断，而非用 FastText 的 “word in wv” 判断。后者可能因为可以合成 OOV 向量而返回真。Word2Vec 对词表外词抛 KeyError；FastText 能返回有限的 100 维向量，但没有直接见过完整词，其近邻也不等于人工语义验证。')

page('4. 组内／组间相似度与实现核对')
table(['预先指定词组','词语'],[[c,'、'.join(ws)] for c,ws in GROUPS.items()],[100,380])
p('所有 16 个词在两个模型的词表中。每组 4 个词，组内取不重复且不含自身的 6 对，四组合计 24 对；跨组每两组 16 对，合计 96 对。计算各词对余弦后平均，不先对词向量求均值。词组在模型结果生成前写入实验配置，未按结果调整。')
table(['模型','组内均值','跨组均值','两者差值'],[[name,f"{e['within_mean']:.4f}",f"{e['between_mean']:.4f}",f"{e['gap']:.4f}"] for name,e in emb.items()],[120,120,120,120])
image('word_group_similarity.png','图 1：各组平均余弦；对角线为组内不同词，非对角线为跨组。')
p('解释时应同时看组内与跨组值：如果两者同时升高，仅凭组内值不能断言分类边界更清晰。这 16 个词属于小规模诊断，没有人工相似度标注，不能代替标准词义评测，也不能据此预判下游准确率。')
h('直接复算 Gensim 的已保存向量')
p('词表内表示 =（词行向量 + 所有字符桶行向量之和）／（字符片段数 + 1）；OOV 表示 = 字符桶行向量的平均值。按照 ft_ngram_hashes 得到的桶索引独立求和，核对足球、计算机及上述 3 个 OOV 词。最大分量误差为 '+f"{max(v['max_absolute_error'] for v in emb['fasttext']['subword_reconstruction'].values()):.2e}"+'，全部通过 1e-5 阈值。该核对验证向量组合公式，没有独立重写并验证哈希函数。')

page('5. 文本分类：有无预训练向量')
p('分类器统一设置：dim=100、lr=0.3、epoch=25、wordNgrams=2、loss=softmax、thread=1、bucket=200000、minn=maxn=0。分类器平均聚合词及词二元组特征，再学习类别权重。基础对照只改变 pretrainedVectors；其他数据、顺序和超参数一致。所有超参数在测试结果产生前固定，没有按测试集调参。')
table(['类别数／训练／测试','从零：准确率／宏 F1','FastText 初始化：准确率／宏 F1'],[[f'{k}／{k*48}／{k*12}',score(run(k,'scratch')),score(run(k,'fasttext'))] for k in [2,4,8]],[125,160,195])
p('上表为 seed=42。准确率是预测正确文章数除以测试文章数；宏 F1 先逐类计算 2PR／(P+R)，再对类别等权平均，无预测或分母为零时记 0。平衡测试集的多数类基线分别为 0.500、0.250、0.125。',small=True)
image('classification_scores.png','图 2：seed=42、43、44 的均值和样本标准差；固定训练／测试划分。')
table(['类别数','从零：宏 F1 均值 ± SD','FastText：宏 F1 均值 ± SD'],[[k,pm(summ(k,'scratch'),'macro_f1'),pm(summ(k,'fasttext'),'macro_f1')] for k in [2,4,8]],[80,200,200])
a=run(8,'fasttext');b=run(8,'scratch')
p(f"8 类 seed=42 时，预训练为 {a['correct']}/96，从零为 {b['correct']}/96，相差 {a['correct']-b['correct']} 篇。配对 bootstrap（10,000 次）准确率差的 95% 区间为 [{paired['paired_bootstrap_accuracy_difference_ci95'][0]:.3f}, {paired['paired_bootstrap_accuracy_difference_ci95'][1]:.3f}]。区间条件于这批弱标签和固定模型，不反映标签正确性、划分不确定性或跨语料泛化。")

page('6. 混淆矩阵与逐类表现')
image('classification_confusion_8.png','图 3：行是标题规则标签，列是模型预测；两图使用同一批 96 篇正文。')
table(['类别','从零 F1','FastText F1','FastText 正确／测试'],[[c,f"{b['classification_report'][c]['f1-score']:.3f}",f"{a['classification_report'][c]['f1-score']:.3f}",f"{a['confusion_matrix'][i][i]}/12"] for i,c in enumerate(CLASSES)],[120,110,120,130])
cm=a['confusion_matrix']
confusions=sorted([(cm[i][j],CLASSES[i],CLASSES[j]) for i in range(8) for j in range(8) if i!=j and cm[i][j]],reverse=True)
p('预训练模型最常见的非对角项：'+'；'.join(f'{x}→{y}：{n} 篇' for n,x,y in confusions[:4])+'。每类仅 12 篇测试文章，一篇就影响本类召回率约 8.3 个百分点，因此逐类数值不宜过度解释。')
p('同样 seed=42 再训练一次 8 类预训练分类器，全部 96 篇的预测标签与原运行一致。另两个种子改变的是分类优化随机性；词向量预训练及文章划分保持固定，因此三次运行的标准差不能解释为整个实验的总体误差。')

page('7. 错误分析与弱标签边界')
errors=[p for p in preds if p['classes']==8 and p['train_per_class']==48 and p['method']=='fasttext' and p['seed']==42 and not p['repeat'] and p['true']!=p['predicted']]
errors.sort(key=lambda x:-x['probability'])
p(f'8 类预训练模型相对规则标签共错 {len(errors)} 篇。下表展示置信度较高的前 6 篇错误；置信度只是模型 softmax 输出，不是经过校准的正确概率。标题仅用于事后分析，不输入分类器。')
table(['文章 ID','规则→预测','标题'],[[r['id'],r['true']+'→'+r['predicted'],r['title']] for r in errors[:6]],[116,90,274])
h('正文证据')
case_rows=[]
for r in errors[:4]:
    text=''.join(articles[r['id']]['body'][:500])
    truth=[w for w in KEYWORDS[r['true']] if w in text]
    guessed=[w for w in KEYWORDS[r['predicted']] if w in text]
    case_rows.append([r['id'],'、'.join(truth) or '未命中','、'.join(guessed) or '未命中'])
table(['文章 ID','正文含规则类别关键词','正文含预测类别关键词'],case_rows,[116,182,182])
p('这些词仅作为透明的诊断线索，不是模型归因，也不能证明预测标签更正确。新闻正文可以同时涉及政策、机构、技术与具体行业，单一类别规则会压缩真实主题。case study 分析的是“与规则标签不一致”，不能把每一处都认定为语义错误。')
h('筛选阶段已经存在的边界例子')
p('19980529-08-005 的标题包含“北京大学登山队”，因“大学”被标成教育，但也可从体育角度理解；19980528-02-002 涉及“科技兴农”，标题规则给科技，正文也可能包含农业。即使排除了同时命中多个主题关键词的标题，跨领域报道依然存在。此处如实保留弱标签，不在看到模型错误后改标或删除测试文章。')

page('8. 扩展：初始化类型与训练规模')
h('Word2Vec 初始化对照（8 类）')
table(['初始化','seed=42 准确率／宏 F1','3 种子宏 F1 均值 ± SD'],[[m,score(run(8,m)),pm(summ(8,m),'macro_f1')] for m in ['scratch','word2vec','fasttext']],[110,180,190])
p('Word2Vec 与 FastText 使用完全相同的预训练句子、词表门槛、维度和监督训练配置。这个对照比只看词组余弦更直接地检验两种表示对当前分类任务的作用。有限的数据、种子与单一超参数配置，仍不足以建立通用模型排名。')
ftpred=[x for x in preds if x['classes']==8 and x['method']=='fasttext' and x['seed']==42 and x['train_per_class']==48 and not x['repeat']]
w2pred=[x for x in preds if x['classes']==8 and x['method']=='word2vec' and x['seed']==42 and x['train_per_class']==48 and not x['repeat']]
same=sum(x['predicted']==y['predicted'] for x,y in zip(ftpred,w2pred))
maxprob=max(abs(x['probability']-y['probability']) for x,y in zip(ftpred,w2pred))
p(f'seed=42 时，两种预训练初始化在 96 篇测试文章中有 {same} 篇标签预测一致，概率最大相差 {maxprob:.3f}；两份 .vec 的 SHA-256 不同，预测文件独立生成。结合三个种子的结果，本实验不足以证明 FastText 初始化有稳定的最终精度优势。',small=True)
h('嵌套小样本学习曲线（8 类）')
image('learning_curve.png','图 4：每类训练 12、24、48 篇；测试始终为同一批 96 篇。')
table(['每类训练／总训练','从零：宏 F1 均值 ± SD','FastText：宏 F1 均值 ± SD'],[[f'{n}／{n*8}',pm(summ(8,'scratch',n),'macro_f1'),pm(summ(8,'fasttext',n),'macro_f1')] for n in [12,24,48]],[120,180,180])
p('小训练集按 SHA256(learning42:文章ID) 排序截取，是 48 篇集合的嵌套子集。这里仅缩减监督样本数：词向量使用的无标签语料保持不变，且其中含有未进入小训练子集的非测试文章。因此曲线衡量固定无标签资源下的标注样本效率，不是预训练语料规模 scaling 实验。')

page('9. 追加诊断：从零训练是否收敛不足')
p('在查看主实验后，从零训练的低分引出了优化轮次不足的可能。于是增加了 100 和 200 轮的诊断：两种初始化使用同一训练／测试集、同一学习率和三个种子，只改变轮次。此项属于事后敏感性分析；不替换 25 轮主表，也不据测试分数选择“最优模型”。若要选择超参数，仍需要另设验证集。')
diagnostic=read_json(RESULTS/'epoch_diagnostic.json')
def ds(epoch,method):
    return next(s for s in diagnostic['summary'] if s['epoch']==epoch and s['method']==method)
table(['轮次','从零：准确率／宏 F1 均值','FastText：准确率／宏 F1 均值'],[[epoch,f"{ds(epoch,'scratch')['accuracy_mean']:.3f} / {ds(epoch,'scratch')['macro_f1_mean']:.3f}",f"{ds(epoch,'fasttext')['accuracy_mean']:.3f} / {ds(epoch,'fasttext')['macro_f1_mean']:.3f}"] for epoch in [25,100,200]],[70,205,205])
table(['轮次','从零：宏 F1 均值 ± SD','FastText：宏 F1 均值 ± SD'],[[epoch,pm(ds(epoch,'scratch'),'macro_f1'),pm(ds(epoch,'fasttext'),'macro_f1')] for epoch in [25,100,200]],[70,205,205])
p('从零训练增加轮次后恢复了大部分表现，说明 25 轮主表中的巨大差距包含收敛速度与优化预算的影响。因而不能把它全部解释为从零训练最终能力的上限，也不能声称预训练在充分训练后必然更准确。相同轮次并不意味着两种初始化都已经充分收敛。')
p('这项诊断也解释了为什么不能只提交一个高分表：预训练表示同时影响初始化位置、收敛速度和有限标注下的优化。学习曲线固定 25 轮，因而同样应理解为固定训练预算下的样本效率。更严格的对照会为各方法分别在独立验证集上选择学习率与轮次，再仅一次评估独立测试集。')
p('各次轮次诊断的指标与 1,152 条逐篇预测保存在 epoch_diagnostic.json，准确率和宏 F1 均已从预测重新核验。主实验 34 次训练、扩展诊断新增 12 次训练，共实际执行 46 次监督模型训练。')

page('10. 结论、局限与复现')
delta=summ(8,'fasttext')['macro_f1_mean']-summ(8,'scratch')['macro_f1_mean']
p(f"固定 25 轮时，8 类 FastText 初始化的宏 F1 三种子均值为 {summ(8,'fasttext')['macro_f1_mean']:.3f}，从零训练为 {summ(8,'scratch')['macro_f1_mean']:.3f}，差为 {delta:.3f}。该结果支持预训练在当前优化预算下的价值。追加的 100／200 轮诊断显示从零训练大幅改善，因此结论应限定于训练预算，不能宣称预训练有普遍的最终精度优势。")
p(f"预训练词表覆盖实际分类正文的训练词次 {coverage['train']['token_coverage']:.2%}、测试词次 {coverage['test']['token_coverage']:.2%}；测试词型覆盖 {coverage['test']['type_coverage']:.2%}。分类阶段关闭字符子词，所以 Gensim 的 OOV 合成能力并没有通过 .vec 一并迁移。官方 fastText 载入预训练文件还会扩充分类词典，字典大小记录在每次运行结果中；这属于通常的预训练导入机制，不能解释成只替换固定词典中的矩阵。")
p('局限：标题关键词产生选择偏差与标签噪声；仅 96 篇固定测试样本；未做全量人工标注、跨年份验证或多次独立划分；标题按 001 行近似界定，另行副标题可能留在正文；正文前 500 词会截断信息；完全正文去重未排除近重复报道；两阶段共享同领域的非测试文章。后续可扩充人工标签，加入验证集、多划分统计与跨领域测试，并在相同协议下比较 Doc2Vec 或词向量平均池化分类器。')
h('复现与验证')
p(f"自动核验状态：{verification['status']}。已重算 {verification['checked_runs']} 次训练的全部准确率、宏 F1 和混淆矩阵，核对 {verification['checked_predictions']:,} 条逐篇预测，并检查正文哈希、测试文章排除以及类别／训练规模嵌套关系。")
code('python3.12 -m venv .venv\nsource .venv/bin/activate\npip install -r requirements-lock.txt\nbash run_all.sh')
p('源 ZIP SHA-256：'+stats['archive_sha256'],small=True)
p('源仓库固定 commit：4020ecc558c554c6f3c9bd81a606d2197e28e628。环境为 Python 3.12.14、Gensim 4.3.3、fasttext-wheel 0.9.2（导入 fasttext）、NumPy 1.26.4、SciPy 1.13.1、scikit-learn 1.5.2。跨平台编译器与浮点实现可能影响末位数值。轻量提交包不含原始全文与大型模型，脚本会下载校验语料并重建。',small=True)
h('参考资料')
references=[
    ('数据镜像与标注说明','https://github.com/chenhui-bupt/PeopleDaily1998'),
    ('Mikolov et al., 2013. Efficient Estimation of Word Representations','https://arxiv.org/abs/1301.3781'),
    ('Bojanowski et al., 2017. Enriching Word Vectors with Subword Information','https://aclanthology.org/Q17-1010/'),
    ('Joulin et al., 2017. Bag of Tricks for Efficient Text Classification','https://aclanthology.org/E17-2068/'),
    ('Gensim FastText 官方文档','https://radimrehurek.com/gensim/models/fasttext.html'),
    ('官方 fastText Python 接口','https://fasttext.cc/docs/en/python-module.html')]
for title,url in references:
    story.append(Paragraph(f'<link href="{url}" color="#176B75">{esc(title)}</link>',SMALL));md.append(f'- [{title}]({url})')

def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#CBDADD'));canvas.line(42,39,553,39)
    canvas.setFont(FONT,8);canvas.setFillColor(colors.HexColor('#60717C'))
    canvas.drawString(42,26,'Gensim / fastText | 可复现实验报告 | 2026-10-07')
    canvas.drawRightString(553,26,str(doc.page));canvas.restoreState()
doc=SimpleDocTemplate(str(ROOT/'report.pdf'),pagesize=(595.28,841.89),leftMargin=57.5,rightMargin=57.5,topMargin=42,bottomMargin=53,
                      title='Gensim 词表示学习与 fastText 文本分类',author='')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(ROOT/'report.md').write_text('\n'.join(md),encoding='utf-8')
print('Created report.pdf and report.md',flush=True)
