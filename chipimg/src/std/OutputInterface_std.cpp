#include "OutputInterface_std.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <limits>
#include <iostream>

using namespace cv;
using std::vector;

static constexpr double kLowPct        = 0.0046;
static constexpr double kHighPct       = 0.0087;
static constexpr double kGamma         = 1.33;
static constexpr bool   kDoOtsu        = true;
static constexpr double kOtsuScale     = 1.0;

static constexpr int    kAreaMin       = 100;
static constexpr int    kAreaMax       = 500;
static constexpr float  kEPS_L1        = 30.f;
static constexpr float  kEPS_L2        = 100.f;

static constexpr float  kRowGapEps     = 25.f;
static constexpr float  kRowAlignYTol  = 10.f;

static constexpr float  kSameColDx     = 30.f;

static constexpr float  kShift_Col2    =  90.f;
static constexpr float  kShift_Col1    =  90.f;

struct DSU {
    vector<int> p, r;
    explicit DSU(int n): p(n), r(n, 0) { std::iota(p.begin(), p.end(), 0); }
    int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (r[a] < r[b]) std::swap(a, b);
        p[b] = a;
        if (r[a] == r[b]) r[a]++;
    }
};

static inline float median1(vector<float> v){
    if (v.empty()) return std::numeric_limits<float>::quiet_NaN();
    std::nth_element(v.begin(), v.begin()+v.size()/2, v.end());
    return v[v.size()/2];
}

static vector<vector<int>> clusterByEpsGroups(const vector<Point2f>& pts, float eps){
    vector<vector<int>> groups; int N=(int)pts.size(); if(!N) return groups;
    DSU d(N); float e2=eps*eps;
    for(int i=0;i<N;++i) for(int j=i+1;j<N;++j){
        Point2f dxy=pts[i]-pts[j];
        if(dxy.x*dxy.x+dxy.y*dxy.y<=e2) d.unite(i,j);
    }
    std::unordered_map<int, vector<int>> m; m.reserve(N*2);
    for(int i=0;i<N;++i) m[d.find(i)].push_back(i);
    groups.reserve(m.size());
    for(auto &kv:m) groups.push_back(std::move(kv.second));
    return groups;
}

static vector<Point2f> groupsToCenters(const vector<Point2f>& pts, const vector<vector<int>>& g){
    vector<Point2f> c; c.reserve(g.size());
    for(const auto &v:g){
        double sx=0,sy=0; for(int id:v){ sx+=pts[id].x; sy+=pts[id].y; }
        float inv=1.f/(float)v.size(); c.emplace_back((float)(sx*inv),(float)(sy*inv));
    }
    return c;
}

static void findPercentile16U(const Mat& img16, double low_pct, double high_pct,
                              uint16_t& low_v, uint16_t& high_v){
    static const int BINS=65536;
    vector<uint32_t> hist(BINS,0);
    for(int r=0;r<img16.rows;++r){
        const uint16_t* p=img16.ptr<uint16_t>(r);
        for(int c=0;c<img16.cols;++c) hist[p[c]]++;
    }
    long long total=1LL*img16.rows*img16.cols;
    long long tl=llround(total*low_pct), th=llround(total*high_pct);
    long long acc=0; int i=0;
    for(;i<BINS;++i){ acc+=hist[i]; if(acc>=tl) break; } low_v=(uint16_t)i;
    acc=0;
    for(i=BINS-1;i>=0;--i){ acc+=hist[i]; if(acc>=th) break; } high_v=(uint16_t)i;
    if(low_v>=high_v){ low_v=0; high_v=65535; }
}

static Mat stretch16U(const Mat& src16, uint16_t a, uint16_t b){
    if(a>=b) return src16.clone();
    Mat f,dst16; src16.convertTo(f,CV_32F);
    f=(f-(float)a)*(65535.f/(float)(b-a));
    threshold(f,f,65535.0,65535.0,THRESH_TRUNC);
    threshold(f,f,0.0,0.0,THRESH_TOZERO);
    f.convertTo(dst16,CV_16U);
    return dst16;
}

static Mat gamma16U(const Mat& src16, float gamma){
    Mat f; src16.convertTo(f,CV_32F,1.0/65535.0);
    pow(f,gamma,f); Mat out; f.convertTo(out,CV_16U,65535.0); return out;
}

static vector<int> assignColsByX(const vector<Point2f>& pts){
    int N=(int)pts.size(); vector<int> col(N,0); if(N<=1) return col;
    vector<int> id(N); std::iota(id.begin(),id.end(),0);
    std::sort(id.begin(),id.end(),[&](int a,int b){ return pts[a].x<pts[b].x; });
    float best_gap=-1.f; int best_pos=-1;
    for(int i=0;i<N-1;++i){
        float gap=pts[id[i+1]].x-pts[id[i]].x;
        if(gap>best_gap){ best_gap=gap; best_pos=i; }
    }
    if(best_pos<0) return col;
    for(int i=0;i<=best_pos;++i) col[id[i]]=0;
    for(int i=best_pos+1;i<N;++i) col[id[i]]=1;
    return col;
}

static vector<int> assignRowsByY(const vector<Point2f>& pts,const vector<int>& col){
    int N=(int)pts.size(); vector<int> row(N,0);
    for(int c=0;c<PointCol;++c){
        vector<int> idc; for(int i=0;i<N;++i) if(col[i]==c) idc.push_back(i);
        if(idc.empty()) continue;
        std::sort(idc.begin(),idc.end(),[&](int a,int b){ return pts[a].y<pts[b].y; });
        int cr=0; float last_y=pts[idc.front()].y; row[idc.front()]=cr;
        for(size_t k=1;k<idc.size();++k){ int i=idc[k]; float y=pts[i].y;
            if((y-last_y)>kRowGapEps && cr+1<PointRow) cr++;
            row[i]=cr; last_y=y;
        }
        for(int i:idc) if(row[i]>=PointRow) row[i]=PointRow-1;
    }
    return row;
}

static int pickAnchorThirdCol(const vector<Point2f>& pts,
                              const vector<int>& col, const vector<int>& row)
{
    int target_col = PointCol - 1;
    auto better = [&](int a, int b){
        if (col[a] != col[b]) return col[a] > col[b];
        if (row[a] != row[b]) return row[a] > row[b];
        if (pts[a].x != pts[b].x) return pts[a].x > pts[b].x;
        return pts[a].y > pts[b].y;
    };
    int best_in_third = -1;
    for (int i=0; i<(int)pts.size(); ++i){
        if (col[i] == target_col){
            if (best_in_third < 0 || better(i, best_in_third)) best_in_third = i;
        }
    }
    if (best_in_third >= 0) return best_in_third;
    int best = -1;
    for (int i=0; i<(int)pts.size(); ++i){
        if (best < 0 || better(i, best)) best = i;
    }
    return best;
}

static bool median_excl(const vector<float>& v,int self,float &out){
    if((int)v.size()<=1) return false; vector<float> t; t.reserve(v.size()-1);
    for(int i=0;i<(int)v.size();++i){ if(i==self) continue; if(std::isfinite(v[i])) t.push_back(v[i]); }
    if(t.empty()) return false; std::nth_element(t.begin(),t.begin()+t.size()/2,t.end()); out=t[t.size()/2]; return true;
}
static bool mean_excl(const vector<float>& v,int self,float &out){
    if((int)v.size()<=1) return false; double s=0; int n=0;
    for(int i=0;i<(int)v.size();++i){ if(i==self) continue; if(std::isfinite(v[i])){ s+=v[i]; n++; } }
    if(!n) return false; out=(float)(s/n); return true;
}
static void fixAnchorsRowOnly(vector<Point2f>& anchors,
                              const vector<Point2f>& centers_l2){
    vector<int> id(centers_l2.size()); std::iota(id.begin(),id.end(),0);
    std::sort(id.begin(),id.end(),[&](int a,int b){
        const auto&A=centers_l2[a],&B=centers_l2[b];
        return (A.y==B.y)?(A.x<B.x):(A.y<B.y);
    });
    int half=(int)id.size()/2;
    vector<int> row0(id.begin(),id.begin()+half), row1(id.begin()+half,id.end());
    auto sortx=[&](vector<int>& t){
        std::sort(t.begin(),t.end(),[&](int a,int b){
            const auto& A = centers_l2[a];
            const auto& B = centers_l2[b];
            return (A.x == B.x) ? (A.y < B.y) : (A.x < B.x);
        });
    };
    sortx(row0); sortx(row1);

    auto process_row = [&](const vector<int>& row_idx){
        if(row_idx.size()<=1) return;
        vector<float> yvals; yvals.reserve(row_idx.size());
        for(size_t k=0;k<row_idx.size();++k) yvals.push_back(anchors[row_idx[k]].y);
        for(size_t k=0;k<row_idx.size();++k){
            int aidx=row_idx[k]; float selfy=anchors[aidx].y, med=0.f;
            if(!median_excl(yvals,(int)k,med)) continue;
            if(!std::isfinite(selfy)||!std::isfinite(med)) continue;
            if(std::fabs(selfy-med)<=kRowAlignYTol) continue;
            float mean_y=0.f; if(mean_excl(yvals,(int)k,mean_y)&&std::isfinite(mean_y)) anchors[aidx].y=mean_y;
        }
    };
    process_row(row0); process_row(row1);
}

struct Grid { Point2f p[PointRow][PointCol]; bool m[PointRow][PointCol]; };

static float estimate_row_step(const vector<Point2f>& col_pts, const vector<Point2f>& all_pts){
    vector<float> d;
    if(col_pts.size()>=2){
        vector<float> ys; ys.reserve(col_pts.size());
        for(auto&p:col_pts) ys.push_back(p.y);
        std::sort(ys.begin(),ys.end());
        for(size_t i=1;i<ys.size();++i) d.push_back(ys[i]-ys[i-1]);
    }
    if(d.empty()){
        vector<float> ys; ys.reserve(all_pts.size());
        for(auto&p:all_pts) ys.push_back(p.y);
        std::sort(ys.begin(),ys.end());
        for(size_t i=1;i<ys.size();++i) d.push_back(ys[i]-ys[i-1]);
    }
    float s=median1(d); if(!std::isfinite(s)||s<=0.f) s=25.f; return s;
}
static inline int row_from_y(float y,float y_anchor,float step){
    float r = (float)(PointRow-1) - (y_anchor - y)/step;
    int   ri = (int)lroundf(r);
    return std::max(0,std::min(PointRow-1,ri));
}

static Grid fitGridAddLeft(const vector<Point2f>& pts, const Point2f& anchor){
    Grid out{};
    for(int r=0;r<PointRow;++r) for(int c=0;c<PointCol;++c){
        out.p[r][c] = Point2f(std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN());
        out.m[r][c] = false;
    }
    if (pts.empty() || !std::isfinite(anchor.y)) return out;

    vector<Point2f> old0, old1;
    for (auto &p: pts){
        if (std::fabs(p.x - anchor.x) <= kSameColDx) old1.push_back(p);
        else                                         old0.push_back(p);
    }

    auto col_median_x = [&](const vector<Point2f>& v)->float{
        if(v.empty()) return std::numeric_limits<float>::quiet_NaN();
        vector<float> xs; xs.reserve(v.size());
        for (auto &p:v) xs.push_back(p.x);
        std::nth_element(xs.begin(), xs.begin()+xs.size()/2, xs.end());
        return xs[xs.size()/2];
    };
    float x_old1 = col_median_x(old1);
    float x_old0 = col_median_x(old0);

    const vector<Point2f>& base_col = (old1.size() >= old0.size()) ? old1 : old0;
    float step = estimate_row_step(base_col, pts);
    if (!std::isfinite(step) || step <= 0.f) step = 25.f;

    auto place = [&](const vector<Point2f>& v, int newC){
        for (auto &p: v){
            int r = row_from_y(p.y, anchor.y, step);
            if (!out.m[r][newC]) { out.p[r][newC] = p; out.m[r][newC] = true; }
            else{
                float ytar = anchor.y - ((float)(PointRow-1 - r))*step;
                float oldd = std::fabs(out.p[r][newC].y - ytar);
                float newd = std::fabs(p.y - ytar);
                if (newd < oldd) out.p[r][newC] = p;
            }
        }
    };
    if (PointCol >= 3){
        place(old0, 1);
        place(old1, 2);
    }else{
        place(old0, 0);
        if (PointCol >= 2) place(old1, 1);
    }

    float x_new2 = std::isfinite(x_old1) ? x_old1 : anchor.x;
    float x_new1 = std::isfinite(x_old0) ? x_old0 : (x_new2 - kShift_Col2);
    float x_new0 = x_new1 - kShift_Col1;

    auto fill_col = [&](int c, float xc){
        if (!std::isfinite(xc)) return;
        for (int r=0; r<PointRow; ++r){
            if (!out.m[r][c]){
                float y = anchor.y - ((float)(PointRow-1 - r))*step;
                out.p[r][c] = Point2f(xc, y);
            }
        }
    };
    if (PointCol >= 3){
        fill_col(0, x_new0);
        fill_col(1, x_new1);
        fill_col(2, x_new2);
    }else{
        fill_col(0, x_new1);
        if (PointCol >= 2) fill_col(1, x_new2);
    }

    return out;
}

static vector<int> sortWellIndex2x8(const vector<Point2f>& centers_l2){
    int N=(int)centers_l2.size();
    vector<int> id(N); std::iota(id.begin(),id.end(),0);
    std::sort(id.begin(),id.end(),[&](int a,int b){
        const auto&A=centers_l2[a],&B=centers_l2[b];
        return (A.y==B.y)?(A.x<B.x):(A.y<B.y);
    });
    int half=N/2; vector<int> row0(id.begin(),id.begin()+half), row1(id.begin()+half,id.end());
    auto sortx=[&](vector<int>& t){
        std::sort(t.begin(),t.end(),[&](int a,int b){
            const auto& A = centers_l2[a];
            const auto& B = centers_l2[b];
            return (A.x == B.x) ? (A.y < B.y) : (A.x < B.x);
        });
    };
    sortx(row0); sortx(row1);
    vector<int> order; order.reserve(N);
    order.insert(order.end(), row0.begin(), row0.end());
    order.insert(order.end(), row1.begin(), row1.end());
    return order;
}

static void CoreDetect(
    const Mat& src16,
    _POINTPOSITIONINFO (&PostionArray)[WellRow][WellCol][PointRow][PointCol])
{
    uint16_t a=0,b=0; findPercentile16U(src16, kLowPct, kHighPct, a, b);
    Mat stretched16 = stretch16U(src16, a, b);
    Mat enhanced16  = gamma16U(stretched16, (float)kGamma);
    Mat view8; enhanced16.convertTo(view8, CV_8U, 1.0/256.0);

    Mat bin8;
    if(kDoOtsu){
        double otsu_th = threshold(view8, bin8, 0, 255, THRESH_BINARY|THRESH_OTSU);
        if(kOtsuScale!=1.0){
            otsu_th = std::max(0.0, std::min(255.0, otsu_th*kOtsuScale));
            threshold(view8, bin8, otsu_th, 255, THRESH_BINARY);
        }
    }else{
        threshold(view8, bin8, 128, 255, THRESH_BINARY);
    }

    Mat labels, stats, centroids;
    int nLabels = connectedComponentsWithStats(bin8, labels, stats, centroids, 8, CV_32S);
    struct Region { Point2f c; int area; };
    vector<Region> regions; regions.reserve(std::max(0, nLabels-1));
    for(int lbl=1; lbl<nLabels; ++lbl){
        int area = stats.at<int>(lbl, CC_STAT_AREA);
        if(area<kAreaMin || area>kAreaMax) continue;
        float cx=(float)centroids.at<double>(lbl,0);
        float cy=(float)centroids.at<double>(lbl,1);
        regions.push_back({Point2f(cx,cy), area});
    }

    vector<Point2f> centers_l1_in; centers_l1_in.reserve(regions.size());
    for(auto&r:regions) centers_l1_in.push_back(r.c);
    auto l1_groups      = clusterByEpsGroups(centers_l1_in, kEPS_L1);
    auto centers_l1_out = groupsToCenters(centers_l1_in, l1_groups);
    auto l2_groups      = clusterByEpsGroups(centers_l1_out, kEPS_L2);
    auto centers_l2     = groupsToCenters(centers_l1_out, l2_groups);

    vector<Point2f> anchors; anchors.reserve(l2_groups.size());
    vector<vector<Point2f>> l2_pts(l2_groups.size());
    for(size_t gi=0; gi<l2_groups.size(); ++gi){
        vector<Point2f> pts;
        for(int id_l1c : l2_groups[gi]){
            for(int raw_idx : l1_groups[id_l1c]){
                pts.push_back(centers_l1_in[raw_idx]);
            }
        }
        if(pts.empty()){ anchors.push_back(Point2f(NAN,NAN)); l2_pts[gi]=std::move(pts); continue; }
        auto col = assignColsByX(pts);
        auto row = assignRowsByY(pts, col);
        int aidx = pickAnchorThirdCol(pts, col, row);
        anchors.push_back(pts[aidx]);
        l2_pts[gi] = std::move(pts);
    }
    fixAnchorsRowOnly(anchors, centers_l2);

    vector<int> order = sortWellIndex2x8(centers_l2);

    for(int wr=0; wr<WellRow; ++wr)
        for(int wc=0; wc<WellCol; ++wc)
            for(int pr=0; pr<PointRow; ++pr)
                for(int pc=0; pc<PointCol; ++pc)
                    PostionArray[wr][wc][pr][pc] = _POINTPOSITIONINFO{};

    int wells = std::min((int)order.size(), WellRow*WellCol);
    for(int k=0; k<wells; ++k){
        int wr = (k < WellCol)? 0 : 1;
        int wc = (k < WellCol)? k : (k-WellCol);
        int gi = order[k];

        const Point2f& anchor = anchors[gi];
        if(!std::isfinite(anchor.x) || !std::isfinite(anchor.y)) continue;

        auto grid = fitGridAddLeft(l2_pts[gi], anchor);
        for(int pr=0; pr<PointRow; ++pr){
            for(int pc=0; pc<PointCol; ++pc){
                _POINTPOSITIONINFO info;
                info.x = grid.p[pr][pc].x;
                info.y = grid.p[pr][pc].y;
                info.measured = grid.m[pr][pc];
                info.valid = std::isfinite(info.x) && std::isfinite(info.y);
                PostionArray[wr][wc][pr][pc] = info;
            }
        }
    }
}

void PerformShapeDetection(
    ushort usImage[],
    _POINTPOSITIONINFO (&PostionArray)[WellRow][WellCol][PointRow][PointCol])
{
    Mat src16(STD_IMG_H, STD_IMG_W, CV_16UC1, (void*)usImage);
    CoreDetect(src16, PostionArray);
}

void PerformShapeDetectionDyn(
    const ushort* usImage, int width, int height,
    _POINTPOSITIONINFO (&PostionArray)[WellRow][WellCol][PointRow][PointCol])
{
    Mat src16(height, width, CV_16UC1, const_cast<ushort*>(usImage));
    CoreDetect(src16, PostionArray);
}
