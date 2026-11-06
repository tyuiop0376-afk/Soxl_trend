import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title='TrendScope | 추세 전환 분석', page_icon='📈', layout='wide')
st.markdown('''<style> .stApp{background:#0b1120;color:#e8eef9} [data-testid="stMetric"]{background:#162238;border:1px solid #2b3c58;border-radius:12px;padding:14px} h1,h2,h3{color:#eef4ff} </style>''', unsafe_allow_html=True)

@st.cache_data(ttl=300,show_spinner=False)
def load(ticker, interval, period):
    x=yf.download(ticker,period=period,interval=interval,auto_adjust=True,progress=False,threads=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    x=x.rename(columns=str.title).dropna(subset=['Open','High','Low','Close'])
    if x.index.tz is not None: x.index=x.index.tz_convert('America/New_York')
    return x

def ema(x,n):return x.ewm(span=n,adjust=False,min_periods=n).mean()
def rma(x,n):return x.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def cross_up(a,b):return (a>b)&(a.shift(1)<=b.shift(1))
def cross_down(a,b):return (a<b)&(a.shift(1)>=b.shift(1))

def calc(df):
    d=df.copy(); o,h,l,c,v=(d[k].astype(float) for k in ['Open','High','Low','Close','Volume'])
    d['EMA9']=ema(c,9);d['EMA21']=ema(c,21);d['SMA50']=c.rolling(50).mean();d['SMA200']=c.rolling(200).mean()
    d['MACD']=ema(c,12)-ema(c,26);d['MACDsig']=ema(d['MACD'],9)
    prev=c.shift();tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1);atr=rma(tr,10)
    # Supertrend 10,3: trailing bands and direction
    mid=(h+l)/2; ub=(mid+3*atr).to_numpy();lb=(mid-3*atr).to_numpy(); cc=c.to_numpy(); up=np.full(len(d),np.nan);lo=np.full(len(d),np.nan);trend=np.full(len(d),np.nan)
    for i in range(len(d)):
        if not np.isfinite(ub[i]):continue
        if i==0 or not np.isfinite(trend[i-1]):up[i]=ub[i];lo[i]=lb[i];trend[i]=1 if cc[i]>=mid.iloc[i] else -1;continue
        up[i]=ub[i] if ub[i]<up[i-1] or cc[i-1]>up[i-1] else up[i-1]
        lo[i]=lb[i] if lb[i]>lo[i-1] or cc[i-1]<lo[i-1] else lo[i-1]
        trend[i]=1 if cc[i]>up[i-1] else (-1 if cc[i]<lo[i-1] else trend[i-1])
    d['ST']=trend
    # Wilder DMI / ADX
    dmplus=h.diff().where((h.diff()>-l.diff())&(h.diff()>0),0)
    dmminus=(-l.diff()).where((-l.diff()>h.diff())&(-l.diff()>0),0)
    atr14=rma(tr,14);d['DI+']=100*rma(dmplus,14)/atr14.replace(0,np.nan);d['DI-']=100*rma(dmminus,14)/atr14.replace(0,np.nan)
    dx=100*(d['DI+']-d['DI-']).abs()/(d['DI+']+d['DI-']).replace(0,np.nan);d['ADX']=rma(dx,14)
    # Parabolic SAR, acceleration 0.02, max 0.2
    sar=np.full(len(d),np.nan)
    if len(d)>2:
        bull=cc[1]>=cc[0];ep=h.iloc[0] if bull else l.iloc[0];s=l.iloc[0] if bull else h.iloc[0];af=.02
        for i in range(1,len(d)):
            s=s+af*(ep-s)
            if bull:
                s=min(s,l.iloc[i-1],l.iloc[max(i-2,0)])
                if l.iloc[i]<s:bull=False;s=ep;ep=l.iloc[i];af=.02
                elif h.iloc[i]>ep:ep=h.iloc[i];af=min(.2,af+.02)
            else:
                s=max(s,h.iloc[i-1],h.iloc[max(i-2,0)])
                if h.iloc[i]>s:bull=True;s=ep;ep=h.iloc[i];af=.02
                elif l.iloc[i]<ep:ep=l.iloc[i];af=min(.2,af+.02)
            sar[i]=s
    d['SAR']=sar
    # Ichimoku: avoid look-ahead; cloud displayed using values from 26 bars ago
    tenkan=(h.rolling(9).max()+l.rolling(9).min())/2;kijun=(h.rolling(26).max()+l.rolling(26).min())/2
    span_a=((tenkan+kijun)/2).shift(26);span_b=((h.rolling(52).max()+l.rolling(52).min())/2).shift(26)
    d['CloudTop']=pd.concat([span_a,span_b],axis=1).max(axis=1);d['CloudBottom']=pd.concat([span_a,span_b],axis=1).min(axis=1)
    d['DonHigh']=h.shift(1).rolling(20).max();d['DonLow']=l.shift(1).rolling(20).min()
    change=c.diff();gain=rma(change.clip(lower=0),14);loss=rma((-change).clip(lower=0),14)
    d['RSI']=100-100/(1+gain/loss.replace(0,np.nan));d.loc[(loss==0)&(gain>0),'RSI']=100;d.loc[(loss==0)&(gain==0),'RSI']=50
    lowest=l.rolling(14).min();highest=h.rolling(14).max();rawk=100*(c-lowest)/(highest-lowest).replace(0,np.nan)
    d['StochK']=rawk.rolling(3).mean();d['StochD']=d['StochK'].rolling(3).mean()
    ma=c.rolling(20).mean();std=c.rolling(20).std();d['BBUpper']=ma+2*std;d['BBLower']=ma-2*std
    tp=(h+l+c)/3;dev=tp.rolling(20).apply(lambda x:np.mean(np.abs(x-np.mean(x))),raw=True);d['CCI']=(tp-tp.rolling(20).mean())/(.015*dev.replace(0,np.nan))
    money=tp*v;pos=money.where(tp>tp.shift(),0);neg=money.where(tp<tp.shift(),0)
    ratio=pos.rolling(14).sum()/neg.rolling(14).sum().replace(0,np.nan);d['MFI']=100-100/(1+ratio)
    d.loc[(neg.rolling(14).sum()==0)&(pos.rolling(14).sum()>0),'MFI']=100
    med=(h+l)/2;d['AO']=med.rolling(5).mean()-med.rolling(34).mean()
    d['EngulfBull']=(c>o)&(c.shift(1)<o.shift(1))&(o<=c.shift(1))&(c>=o.shift(1))
    d['EngulfBear']=(c<o)&(c.shift(1)>o.shift(1))&(o>=c.shift(1))&(c<=o.shift(1))
    # Session anchored VWAP: meaningful only on intraday bars
    if 'Datetime' in str(d.index.dtype) or isinstance(d.index,pd.DatetimeIndex):
        dates=pd.Series(d.index.date,index=d.index)
        d['VWAP']=(tp*v).groupby(dates).cumsum()/v.groupby(dates).cumsum().replace(0,np.nan)
        d['SessionOpen']=o.groupby(dates).transform('first');d['SessionHigh']=h.groupby(dates).cummax().shift(1)
    else:d['VWAP']=np.nan;d['SessionOpen']=np.nan;d['SessionHigh']=np.nan
    return d

def sig(value,positive,negative):
    if pd.isna(value):return None
    return 1 if positive else (-1 if negative else 0)

def signals(d,interval):
    r=d.iloc[-1];p=d.iloc[-2];c=r.Close
    tests=[
      ('EMA 9·21 교차',sig(c,r.EMA9>r.EMA21,r.EMA9<r.EMA21),'추세',10),
      ('골든·데드크로스 50·200',sig(c,r.SMA50>r.SMA200,r.SMA50<r.SMA200),'추세',6),
      ('MACD 12·26·9',sig(r.MACD,r.MACD>r.MACDsig,r.MACD<r.MACDsig),'추세',10),
      ('슈퍼트렌드 10·3',sig(r.ST,r.ST>0,r.ST<0),'추세',12),
      ('파라볼릭 SAR',sig(r.SAR,c>r.SAR,c<r.SAR),'추세',5),
      ('DMI·ADX 14',sig(r.ADX,r.ADX>=20 and r['DI+']>r['DI-'],r.ADX>=20 and r['DI+']<r['DI-']),'추세',8),
      ('일목균형표 9·26·52',sig(r.CloudTop,c>r.CloudTop,c<r.CloudBottom),'추세',8),
      ('돈치안 돌파 20',sig(r.DonHigh,c>r.DonHigh,c<r.DonLow),'추세',5),
      ('RSI 14',sig(r.RSI,r.RSI>55,r.RSI<45),'과매수·과매도',6),
      ('스토캐스틱 14·3·3',sig(r.StochK,r.StochK>r.StochD and r.StochK<80,r.StochK<r.StochD and r.StochK>20),'과매수·과매도',4),
      ('볼린저 밴드 20·2',sig(r.BBUpper,c>r.BBUpper,c<r.BBLower),'과매수·과매도',3),
      ('CCI 20',sig(r.CCI,r.CCI>100,r.CCI< -100),'과매수·과매도',4),
      ('MFI 14',sig(r.MFI,r.MFI>55,r.MFI<45),'과매수·과매도',4),
      ('어썸 오실레이터',sig(r.AO,r.AO>0,r.AO<0),'과매수·과매도',5),
      ('장악형 캔들',1 if r.EngulfBull else (-1 if r.EngulfBear else 0),'캔들',3),
    ]
    if interval!='1d':
        tests.extend([('VWAP 교차',sig(r.VWAP,c>r.VWAP,c<r.VWAP),'분봉 전용',4),
                      ('시가 범위 돌파 (근사)',sig(r.SessionOpen,c>r.SessionOpen and c>p.High,c<r.SessionOpen and c<p.Low),'분봉 전용',3)])
    else:tests.extend([('VWAP 교차',None,'분봉 전용',4),('시가 범위 돌파 30분',None,'분봉 전용',3)])
    # Do not label approximated opening range as exact: this indicator is disabled until actual 30m data is used
    tests[-1]=(tests[-1][0],None,'분봉 전용',tests[-1][3])
    active=[t for t in tests if t[1] is not None]
    score=round(sum(t[1]*t[3] for t in active)/sum(t[3] for t in active)*100) if active else 0
    return tests,score

st.title('📈 TrendScope')
st.caption('무료 주식 추세 전환 분석 · 교육 및 연구용 · 투자 권유 아님')
with st.sidebar:
    st.header('분석 설정')
    ticker=st.text_input('미국 주식 티커',value='SOXL',max_chars=12).strip().upper()
    if not ticker or not all(ch.isalnum() or ch in '.-' for ch in ticker):st.error('올바른 티커를 입력하세요.');st.stop()
    tf=st.selectbox('차트 시간봉',['1시간봉','4시간봉','일봉'])
    if st.button('🔄 데이터 새로고침',use_container_width=True):load.clear();st.rerun()
    st.caption('무료 Yahoo Finance 데이터는 지연·누락될 수 있습니다. 장중 마지막 봉은 확정 전입니다.')

@st.cache_data(ttl=300,show_spinner=False)
def get_frames(ticker):
    h=load(ticker,'1h','3mo');day=load(ticker,'1d','2y')
    # Yahoo does not consistently provide native 4h US equities; aggregate from regular-session hourly bars.
    if not h.empty:
        local=h.between_time('09:30','15:59').copy()
        if not local.empty:
            # 4-hour bins anchored at 09:30 US/Eastern; 2nd bin is shorter on regular trading days
            mins=local.index.hour*60+local.index.minute
            bucket=np.maximum(0,((mins-570)//240)).astype(int)
            group=pd.MultiIndex.from_arrays([local.index.date,bucket])
            four=local.groupby(group).agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
            idx=[]
            for date,b in four.index:idx.append(pd.Timestamp(date).tz_localize('America/New_York')+pd.Timedelta(minutes=570+int(b)*240))
            four.index=pd.DatetimeIndex(idx)
        else:four=pd.DataFrame()
    else:four=pd.DataFrame()
    return {'1시간봉':h,'4시간봉':four,'일봉':day}

try:frames=get_frames(ticker)
except Exception as e:st.error(f'시세 데이터를 불러오지 못했습니다: {e}');st.stop()
results={}
for label,data in frames.items():
    if len(data)>=60:
        d=calc(data);tests,score=signals(d,'1d' if label=='일봉' else '1h');results[label]=(d,tests,score)
if tf not in results:st.warning('선택한 시간봉의 데이터가 부족합니다. 다른 시간봉을 선택하거나 잠시 후 다시 시도하세요.');st.stop()
d,tests,score=results[tf];last=d.iloc[-1]
status='강한 상승 우세' if score>=60 else '상승 우세' if score>=25 else '강한 하락 우세' if score<=-60 else '하락 우세' if score<=-25 else '중립·혼조'
col1,col2,col3,col4=st.columns(4)
col1.metric('현재가 (USD)',f'${last.Close:,.2f}')
col2.metric('추세 점수',f'{score:+d} / 100')
col3.metric('판정',status)
col4.metric('활성 매수 / 매도',f'{sum(x[1]==1 for x in tests)} / {sum(x[1]==-1 for x in tests)}')
st.caption(f'선택 시간봉 마지막 데이터: {d.index[-1]} · 미국 동부시간 · 장중 미확정 봉 포함 가능')
st.subheader('시간봉별 추세')
cols=st.columns(3)
for col,label in zip(cols,['일봉','4시간봉','1시간봉']):
    if label in results:
        val=results[label][2];col.metric(label,f'{val:+d}점', '상승 우세' if val>=25 else '하락 우세' if val<=-25 else '중립',delta_color='off')
    else:col.metric(label,'데이터 부족')
st.caption('4시간봉은 정규장 1시간봉을 합성한 근사치이며, 일봉과 분봉의 계산 기준이 다를 수 있습니다.')

st.subheader('주가 및 이동평균선')
plot=d.tail(130);fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.77,.23],vertical_spacing=.04)
fig.add_trace(go.Candlestick(x=plot.index,open=plot.Open,high=plot.High,low=plot.Low,close=plot.Close,name='가격'),row=1,col=1)
for key,color in [('EMA9','#54d6c6'),('EMA21','#f6b55c'),('SMA50','#8ea7ff'),('SMA200','#d596f8')]:fig.add_trace(go.Scatter(x=plot.index,y=plot[key],mode='lines',name=key,line=dict(color=color,width=1.5)),row=1,col=1)
fig.add_trace(go.Bar(x=plot.index,y=plot.Volume,name='거래량',marker_color='#637ca2'),row=2,col=1)
fig.update_layout(height=580,template='plotly_dark',paper_bgcolor='#0b1120',plot_bgcolor='#0b1120',xaxis_rangeslider_visible=False,legend=dict(orientation='h',y=1.1),margin=dict(l=8,r=8,t=25,b=8))
st.plotly_chart(fig,use_container_width=True)

st.subheader('기술적 지표별 신호')
rows=[]
for name,signal,category,weight in tests:
    rows.append({'구분':category,'지표':name,'현재 신호':'매수' if signal==1 else '매도' if signal==-1 else '중립' if signal==0 else '계산 불가','가중치':weight})
st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
st.caption('점수는 각 지표의 방향성에 대한 가중 합계이며 검증된 승률이나 상승 확률이 아닙니다. 볼린저 밴드는 돌파 방향을 추세 신호로 해석합니다. 시가 범위 돌파 30분은 별도 30분봉 데이터가 필요해 계산에서 제외했습니다.')
st.warning('⚠️ SOXL은 일일 수익률 3배를 추종하는 레버리지 ETF입니다. 변동성·복리 효과·급락 위험이 크며, 신호가 실제 수익을 보장하지 않습니다. 자동 주문 기능은 없습니다.')
with st.expander('데이터 및 계산 방식'):
    st.write('Yahoo Finance 비공식 무료 데이터(yfinance)를 사용합니다. 1시간봉은 최근 약 3개월, 일봉은 최근 2년을 요청합니다. 일부 지표는 과거 데이터가 부족하면 계산 불가로 표시됩니다. 일목균형표 구름은 26봉 이전 값을 이용하여 미래 정보를 사용하지 않습니다. 정규장 기준 4시간봉은 09:30부터 4시간과 잔여 구간으로 구성됩니다. VWAP은 해당 봉 데이터의 일별 누적 거래량 가중 평균이며, 30분 시가 범위 돌파는 계산하지 않습니다. 데이터가 제공되지 않는 시간에는 최신 수치가 갱신되지 않습니다.')
