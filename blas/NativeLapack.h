//
// Created by agibsonccc on 5/28/16.
//

#ifndef LIBND4J_NATIVELAPACK_H_H
#define LIBND4J_NATIVELAPACK_H_H
#include <pointercast.h>


//DO NOT REMOVE: THIS IS AN EDITOR SEMANTICS THING FOR CLION
//IT DEFINES THE EXPORT MACRO FOR THE EDITOR AND THEN
//RE ADDS THE DEFINITION VIA dll.h
#ifdef  _WIN32
#define ND4J_EXPORT __declspec(dllexport)
#else
#define ND4J_EXPORT
#endif
#include <dll.h>

class ND4J_EXPORT Nd4jLapack  {
        int dgbbrd(int matriint, char vect, int m, int n, int ncc, int kl, int ku, long ab, int ldab,
                          long d, long e, long q, int ldq,
                          long pt, int ldpt, long c, int ldc );
        int cgbbrd(int matriint, char vect, int m, int n, int ncc, int kl, int ku, IComplexNumber ab, int ldab, long d, long e,
                          IComplexNumber q, int ldq,
                          IComplexNumber pt, int ldpt,
                          IComplexNumber c, int ldc );
        int zgbbrd(int matriint, char vect, int m,int n, int ncc, int kl,int ku, IComplexNumber ab,int ldab, long d, long e,
                          IComplexNumber q, int ldq,
                          IComplexNumber pt, int ldpt,
                          IComplexNumber c, int ldc );

        int sgbcon(int matriint, char norm, int n,int kl, int ku,  long ab,int ldab,int ipiv, float anorm,
                          long rcond );
        int dgbcon(int matriint, char norm, int n,int kl, int ku,  long ab,int ldab,int ipiv,
                          double anorm, long rcond );
        int cgbcon(int matriint, char norm, int n,int kl, int ku,
                          IComplexNumber ab, int ldab,int ipiv, float anorm, long rcond );
        int zgbcon(int matriint, char norm, int n,int kl, int ku,
                          IComplexNumber ab, int ldab,int ipiv, double anorm,
                          long rcond );

        int sgbequ(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c, long rowcnd,
                          long colcnd, long amax );
        int dgbequ(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c,
                          long rowcnd, long colcnd, long amax );
        int cgbequ(int matriint, int m, int n,int kl, int ku,
                          IComplexNumber ab, int ldab,
                          long r, long c, long rowcnd, long colcnd,
                          long amax );
        int zgbequ(int matriint, int m, int n,int kl, int ku,
                          IComplexNumber ab, int ldab,
                          long r, long c, long rowcnd, long colcnd,
                          long amax );

        int sgbequb(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c, long rowcnd,
                           long colcnd, long amax );
        int dgbequb(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c,
                           long rowcnd, long colcnd, long amax );
        int cgbequb(int matriint, int m, int n,int kl, int ku,
                           IComplexNumber ab, int ldab,
                           long r, long c, long rowcnd, long colcnd,
                           long amax );
        int zgbequb(int matriint, int m, int n,int kl, int ku,
                           IComplexNumber ab, int ldab,
                           long r, long c, long rowcnd,
                           long colcnd, long amax );

        int sgbrfs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                          long ab, int ldab,  long afb,int ldafb,int ipiv,
                          long b, int ldb, long x,int ldx, long ferr, long berr );
        int dgbrfs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                          long ab, int ldab,  long afb,int ldafb,int ipiv,
                          long b, int ldb, long x,int ldx, long ferr, long berr );
        int cgbrfs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                          IComplexNumber ab, int ldab,
                          IComplexNumber afb, int ldafb,int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx, long ferr,
                          long berr );
        int zgbrfs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                          IComplexNumber ab, int ldab,
                          IComplexNumber afb, int ldafb,int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );

        int sgbrfsx(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,  long ab, int ldab,
                           long afb, int ldafb,int ipiv,  long r,
                           long c,  long b, int ldb,
                           long x, int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );
        int dgbrfsx(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,  long ab, int ldab,
                           long afb, int ldafb,int ipiv,  long r,
                           long c,  long b, int ldb,
                           long x, int ldx, long rcond,
                           long berr, int int_bnds,
                           long erint_norm, long erint_comp,int nparams, long params );
        int cgbrfsx(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,  IComplexNumber ab,int ldab,  IComplexNumber afb,int ldafb,int ipiv,
                           long r,  long c,
                           IComplexNumber b, int ldb,
                           IComplexNumber x, int ldx,
                           long rcond, long berr, int int_bnds,
                           long erint_norm, long erint_comp,int nparams, long params );
        int zgbrfsx(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,  IComplexNumber ab,int ldab,  IComplexNumber afb,int ldafb,int ipiv,
                           long r,  long c,
                           IComplexNumber b, int ldb,
                           IComplexNumber x, int ldx,
                           long rcond, long berr, int int_bnds,
                           long erint_norm, long erint_comp,int nparams, long params );

        int sgbsv(int matriint, int n, int kl,int ku, int nrhs, long ab,int ldab, int ipiv, long b,int ldb );
        int dgbsv(int matriint, int n, int kl,int ku, int nrhs, long ab,int ldab, int ipiv, long b,int ldb );
        int cgbsv(int matriint, int n, int kl,int ku, int nrhs,
                         IComplexNumber ab, int ldab,int ipiv, IComplexNumber b,int ldb );
        int zgbsv(int matriint, int n, int kl,int ku, int nrhs,
                         IComplexNumber ab, int ldab,int ipiv, IComplexNumber b,int ldb );

        int sgbsvx(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                          long afb, int ldafb, int ipiv,
                          String equed, long r, long c, long b,int ldb, long x, int ldx,
                          long rcond, long ferr, long berr,
                          long rpivot );
        int dgbsvx(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                          long afb, int ldafb, int ipiv,
                          String equed, long r, long c, long b,int ldb, long x, int ldx,
                          long rcond, long ferr, long berr,
                          long rpivot );
        int cgbsvx(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed,
                          long r, long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                          long berr, long rpivot );
        int zgbsvx(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed,
                          long r, long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                          long berr, long rpivot );

        int sgbsvxx(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                           long afb, int ldafb, int ipiv,
                           String equed, long r, long c, long b,int ldb, long x, int ldx,
                           long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,long params );
        int dgbsvxx(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                           long afb, int ldafb, int ipiv,
                           String equed, long r, long c, long b,int ldb, long x, int ldx,
                           long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );
        int cgbsvxx(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed,
                           long r, long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long rpvgrw,
                           long berr, int int_bnds,
                           long erint_norm, long erint_comp,int nparams, long params );
        int sgbtrf(int matriint, int m, int n,int kl, int ku, long ab,int ldab, int ipiv );
        int dgbtrf(int matriint, int m, int n,int kl, int ku, long ab,int ldab, int ipiv );
        int cgbtrf(int matriint, int m, int n,int kl, int ku,
                          IComplexNumber ab, int ldab,int ipiv );
        int zgbtrf(int matriint, int m, int n,int kl, int ku,
                          IComplexNumber ab, int ldab,int ipiv );

        int sgbtrs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                          long ab, int ldab,int ipiv, long b, int ldb );
        int dgbtrs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                          long ab, int ldab,int ipiv, long b, int ldb );
        int cgbtrs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                          IComplexNumber ab, int ldab,int ipiv, IComplexNumber b,int ldb );
        int zgbtrs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                          IComplexNumber ab, int ldab,int ipiv, IComplexNumber b,int ldb );

        int sgebak(int matriint, char job, char side, int n,int ilo, int ihi,  long scale,int m, long v, int ldv );
        int dgebak(int matriint, char job, char side, int n,int ilo, int ihi,  long scale,int m, long v, int ldv );
        int cgebak(int matriint, char job, char side, int n,int ilo, int ihi,  long scale,int m, IComplexNumber v,int ldv );
        int zgebak(int matriint, char job, char side, int n,int ilo, int ihi,  long scale,int m, IComplexNumber v,int ldv );

        int sgebal(int matriint, char job, int n, long a,int lda, int ilo, int ihi,
                          long scale );
        int dgebal(int matriint, char job, int n, long a,int lda, int ilo, int ihi,
                          long scale );
        int cgebal(int matriint, char job, int n,
                          IComplexNumber a, int lda,int ilo, int ihi, long scale );
        int zgebal(int matriint, char job, int n,
                          IComplexNumber a, int lda,int ilo, int ihi, long scale );

        int sgebrd(int matriint, int m, int n,
                          long a, int lda, long d, long e,
                          long tauq, long taup );
        int dgebrd(int matriint, int m, int n,
                          long a, int lda, long d, long e,
                          long tauq, long taup );
        int cgebrd(int matriint, int m, int n,
                          IComplexNumber a, int lda, long d,
                          long e, IComplexNumber tauq,
                          IComplexNumber taup );
        int zgebrd(int matriint, int m, int n,
                          IComplexNumber a, int lda, long d,
                          long e, IComplexNumber tauq,
                          IComplexNumber taup );

        int sgecon(int matriint, char norm, int n,
                          long a, int lda, float anorm,
                          long rcond );
        int dgecon(int matriint, char norm, int n,
                          long a, int lda, double anorm,
                          long rcond );
        int cgecon(int matriint, char norm, int n,
                          IComplexNumber a, int lda,
                          float anorm, long rcond );
        int zgecon(int matriint, char norm, int n,
                          IComplexNumber a, int lda,
                          double anorm, long rcond );

        int sgeequ(int matriint, int m, int n,
                          long a, int lda, long r, long c,
                          long rowcnd, long colcnd, long amax );
        int dgeequ(int matriint, int m, int n,
                          long a, int lda, long r,
                          long c, long rowcnd, long colcnd,
                          long amax );
        int cgeequ(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          long r, long c, long rowcnd, long colcnd,
                          long amax );
        int zgeequ(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          long r, long c, long rowcnd, long colcnd,
                          long amax );

        int sgeequb(int matriint, int m, int n,
                           long a, int lda, long r, long c,
                           long rowcnd, long colcnd, long amax );
        int dgeequb(int matriint, int m, int n,
                           long a, int lda, long r,
                           long c, long rowcnd, long colcnd,
                           long amax );
        int cgeequb(int matriint, int m, int n,
                           IComplexNumber a, int lda,
                           long r, long c, long rowcnd, long colcnd,
                           long amax );
        int zgeequb(int matriint, int m, int n,
                           IComplexNumber a, int lda,
                           long r, long c, long rowcnd,
                           long colcnd, long amax );

        int sgees(int matriint, char jobvs, char sort,int select, int n, long a,int lda, int sdim, long wr,
                         long wi, long vs, int ldvs );
        int dgees(int matriint, char jobvs, char sort,int select, int n, long a,int lda, int sdim, long wr,
                         long wi, long vs, int ldvs );
        int cgees(int matriint, char jobvs, char sort,int select, int n,
                         IComplexNumber a, int lda,int sdim, IComplexNumber w,
                         IComplexNumber vs, int ldvs );
        int zgees(int matriint, char jobvs, char sort,int select, int n,
                         IComplexNumber a, int lda,int sdim, IComplexNumber w,
                         IComplexNumber vs, int ldvs );

        int sgeesx(int matriint, char jobvs, char sort,int select, char sense, int n,
                          long a, int lda, int sdim,
                          long wr, long wi, long vs, int ldvs,
                          long rconde, long rcondv );
        int dgeesx(int matriint, char jobvs, char sort,int select, char sense, int n,
                          long a, int lda, int sdim,
                          long wr, long wi, long vs, int ldvs,
                          long rconde, long rcondv );
        int cgeesx(int matriint, char jobvs, char sort,int select, char sense, int n,
                          IComplexNumber a, int lda,int sdim, IComplexNumber w,
                          IComplexNumber vs, int ldvs,
                          long rconde, long rcondv );
        int zgeesx(int matriint, char jobvs, char sort,int select, char sense, int n,
                          IComplexNumber a, int lda,int sdim, IComplexNumber w,
                          IComplexNumber vs, int ldvs,
                          long rconde, long rcondv );

        int sgeev(int matriint, char jobvl, char jobvr,int n, long a, int lda, long wr,
                         long wi, long vl, int ldvl, long vr,int ldvr );
        int dgeev(int matriint, char jobvl, char jobvr,int n, long a, int lda, long wr,
                         long wi, long vl, int ldvl, long vr,int ldvr );
        int cgeev(int matriint, char jobvl, char jobvr,int n, IComplexNumber a, int lda,
                         IComplexNumber w, IComplexNumber vl,int ldvl, IComplexNumber vr,int ldvr );
        int zgeev(int matriint, char jobvl, char jobvr,int n, IComplexNumber a,int lda, IComplexNumber w,
                         IComplexNumber vl, int ldvl,
                         IComplexNumber vr, int ldvr );

        int sgeevx(int matriint, char balanc, char jobvl,
                          char jobvr, char sense, int n, long a,int lda, long wr, long wi, long vl,int ldvl, long vr, int ldvr,int ilo, int ihi, long scale,
                          long abnrm, long rconde, long rcondv );
        int dgeevx(int matriint, char balanc, char jobvl,
                          char jobvr, char sense, int n, long a,int lda, long wr, long wi, long vl,int ldvl, long vr, int ldvr,int ilo, int ihi, long scale,
                          long abnrm, long rconde, long rcondv );
        int cgeevx(int matriint, char balanc, char jobvl,
                          char jobvr, char sense, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber w, IComplexNumber vl,int ldvl, IComplexNumber vr,int ldvr, int ilo, int ihi,
                          long scale, long abnrm, long rconde,
                          long rcondv );
        int zgeevx(int matriint, char balanc, char jobvl,
                          char jobvr, char sense, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber w, IComplexNumber vl,int ldvl, IComplexNumber vr,int ldvr, int ilo, int ihi,
                          long scale, long abnrm, long rconde,
                          long rcondv );

        int sgehrd(int matriint, int n, int ilo,int ihi, long a, int lda,
                          long tau );
        int dgehrd(int matriint, int n, int ilo,int ihi, long a, int lda,
                          long tau );
        int cgehrd(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda, IComplexNumber tau );
        int zgehrd(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda, IComplexNumber tau );

        int sgejsv(int matriint, char joba, char jobu, char jobv,
                          char jobr, char jobt, char jobp, int m,int n, long a, int lda, long sva,
                          long u, int ldu, long v, int ldv,
                          long stat, int istat );
        int dgejsv(int matriint, char joba, char jobu, char jobv,
                          char jobr, char jobt, char jobp, int m,int n, long a, int lda, long sva,
                          long u, int ldu, long v, int ldv,
                          long stat, int istat );
        int cgejsv(int matriint, char joba, char jobu, char jobv,
                          char jobr, char jobt, char jobp, int m,int n, IComplexNumber a, int lda, long sva,
                          IComplexNumber u, int ldu, IComplexNumber v, int ldv,
                          long stat, int istat );
        int zgejsv(int matriint, char joba, char jobu, char jobv,
                          char jobr, char jobt, char jobp, int m,int n, IComplexNumber a, int lda, long sva,
                          IComplexNumber u, int ldu, IComplexNumber v, int ldv,
                          long stat, int istat );

        int sgelq2(int matriint, int m, int n,
                          long a, int lda, long tau );
        int dgelq2(int matriint, int m, int n,
                          long a, int lda, long tau );
        int cgelq2(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );
        int zgelq2(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );

        int sgelqf(int matriint, int m, int n,
                          long a, int lda, long tau );
        int dgelqf(int matriint, int m, int n,
                          long a, int lda, long tau );
        int cgelqf(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );
        int zgelqf(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );

        int sgels(int matriint, char trans, int m,int n, int nrhs, long a,int lda, long b, int ldb );
        int dgels(int matriint, char trans, int m,int n, int nrhs, long a,int lda, long b, int ldb );
        int cgels(int matriint, char trans, int m,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb );
        int zgels(int matriint, char trans, int m,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb );

        int sgelsd(int matriint, int m, int n,int nrhs, long a, int lda, long b,int ldb, long s, float rcond,int rank );
        int dgelsd(int matriint, int m, int n,int nrhs, long a, int lda,
                          long b, int ldb, long s, double rcond,int rank );
        int cgelsd(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, float rcond,int rank );
        int zgelsd(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, double rcond,int rank );

        int sgelss(int matriint, int m, int n,int nrhs, long a, int lda, long b,int ldb, long s, float rcond,int rank );
        int dgelss(int matriint, int m, int n,int nrhs, long a, int lda,
                          long b, int ldb, long s, double rcond,int rank );
        int cgelss(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, float rcond,int rank );
        int zgelss(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, double rcond,int rank );

        int sgelsy(int matriint, int m, int n,int nrhs, long a, int lda, long b,int ldb, int jpvt, float rcond,int rank );
        int dgelsy(int matriint, int m, int n,int nrhs, long a, int lda,
                          long b, int ldb, int jpvt,
                          double rcond, int rank );
        int cgelsy(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, int jpvt, float rcond,int rank );
        int zgelsy(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, int jpvt, double rcond,int rank );

        int sgeqlf(int matriint, int m, int n,
                          long a, int lda, long tau );
        int dgeqlf(int matriint, int m, int n,
                          long a, int lda, long tau );
        int cgeqlf(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );
        int zgeqlf(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );

        int sgeqp3(int matriint, int m, int n,
                          long a, int lda, int jpvt,
                          long tau );
        int dgeqp3(int matriint, int m, int n,
                          long a, int lda, int jpvt,
                          long tau );
        int cgeqp3(int matriint, int m, int n,
                          IComplexNumber a, int lda,int jpvt, IComplexNumber tau );
        int zgeqp3(int matriint, int m, int n,
                          IComplexNumber a, int lda,int jpvt, IComplexNumber tau );

        int sgeqpf(int matriint, int m, int n,
                          long a, int lda, int jpvt,
                          long tau );
        int dgeqpf(int matriint, int m, int n,
                          long a, int lda, int jpvt,
                          long tau );
        int cgeqpf(int matriint, int m, int n,
                          IComplexNumber a, int lda,int jpvt, IComplexNumber tau );
        int zgeqpf(int matriint, int m, int n,
                          IComplexNumber a, int lda,int jpvt, IComplexNumber tau );

        int sgeqr2(int matriint, int m, int n,
                          long a, int lda, long tau );
        int dgeqr2(int matriint, int m, int n,
                          long a, int lda, long tau );
        int cgeqr2(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );
        int zgeqr2(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );

        int sgeqrf(int matriint, int m, int n,
                          long a, int lda, long tau );
        int dgeqrf(int matriint, int m, int n,
                          long a, int lda, long tau );
        int cgeqrf(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );
        int zgeqrf(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );

        int sgeqrfp(int matriint, int m, int n,
                           long a, int lda, long tau );
        int dgeqrfp(int matriint, int m, int n,
                           long a, int lda, long tau );
        int cgeqrfp(int matriint, int m, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber tau );
        int zgeqrfp(int matriint, int m, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber tau );

        int sgerfs(int matriint, char trans, int n,int nrhs,  long a, int lda,
                          long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                          long ferr, long berr );
        int dgerfs(int matriint, char trans, int n,int nrhs,  long a, int lda,
                          long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                          long ferr, long berr );
        int cgerfs(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx, long ferr,
                          long berr );
        int zgerfs(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );

        int sgerfsx(int matriint, char trans, char equed,int n, int nrhs,  long a,int lda,  long af, int ldaf,int ipiv,  long r,
                           long c,  long b, int ldb,
                           long x, int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );
        int dgerfsx(int matriint, char trans, char equed,int n, int nrhs,  long a,int lda,  long af, int ldaf,int ipiv,  long r,
                           long c,  long b, int ldb,
                           long x, int ldx, long rcond,
                           long berr, int int_bnds,
                           long erint_norm, long erint_comp,int nparams, long params );
        int cgerfsx(int matriint, char trans, char equed,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,int ipiv,  long r,
                           long c,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );
        int zgerfsx(int matriint, char trans, char equed,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,int ipiv,  long r,
                           long c,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );

        int sgerqf(int matriint, int m, int n,
                          long a, int lda, long tau );
        int dgerqf(int matriint, int m, int n,
                          long a, int lda, long tau );
        int cgerqf(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );
        int zgerqf(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );

        int sgesdd(int matriint, char jobz, int m,int n, long a, int lda, long s,
                          long u, int ldu, long vt,int ldvt );
        int dgesdd(int matriint, char jobz, int m,int n, long a, int lda, long s,
                          long u, int ldu, long vt,int ldvt );
        int cgesdd(int matriint, char jobz, int m,int n, IComplexNumber a,int lda, long s, IComplexNumber u,int ldu, IComplexNumber vt,int ldvt );
        int zgesdd(int matriint, char jobz, int m,int n, IComplexNumber a,int lda, long s, IComplexNumber u,int ldu, IComplexNumber vt,int ldvt );

        int sgesv(int matriint, int n, int nrhs,
                         long a, int lda, int ipiv, long b,int ldb );
        int dgesv(int matriint, int n, int nrhs,
                         long a, int lda, int ipiv,
                         long b, int ldb );
        int cgesv(int matriint, int n, int nrhs,
                         IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb );
        int zgesv(int matriint, int n, int nrhs,
                         IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb );
        int dsgesv(int matriint, int n, int nrhs,
                          long a, int lda, int ipiv,
                          long b, int ldb, long x, int ldx,int iter );
        int zcgesv(int matriint, int n, int nrhs,
                          IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, IComplexNumber x,int ldx, int iter );

        int sgesvd(int matriint, char jobu, char jobvt,int m, int n, long a, int lda,
                          long s, long u, int ldu, long vt,int ldvt, long superb );
        int dgesvd(int matriint, char jobu, char jobvt,int m, int n, long a,int lda, long s, long u, int ldu,
                          long vt, int ldvt, long superb );
        int cgesvd(int matriint, char jobu, char jobvt,int m, int n, IComplexNumber a,int lda, long s, IComplexNumber u,int ldu, IComplexNumber vt,int ldvt, long superb );
        int zgesvd(int matriint, char jobu, char jobvt,int m, int n, IComplexNumber a,int lda, long s, IComplexNumber u,int ldu, IComplexNumber vt,int ldvt, long superb );

        int sgesvdx(int matriint, char jobu, char jobvt, char range,int m, int n, long a,int lda, int vl, int vu,int il, int iu, int ns,
                           long s, long u, int ldu,
                           long vt, int ldvt,int superb );
        int dgesvdx(int matriint, char jobu, char jobvt, char range,int m, int n, long a,int lda, int vl, int vu,int il, int iu, int ns,
                           long s, long u, int ldu,
                           long vt, int ldvt,int superb );
        int cgesvdx(int matriint, char jobu, char jobvt, char range,int m, int n, IComplexNumber a,int lda, int vl, int vu,int il, int iu, int ns,
                           long s, IComplexNumber u, int ldu,
                           IComplexNumber vt, int ldvt,int superb );
        int zgesvdx(int matriint, char jobu, char jobvt, char range,int m, int n, IComplexNumber a,int lda, int vl, int vu,int il, int iu, int ns,
                           long s, IComplexNumber u, int ldu,
                           IComplexNumber vt, int ldvt,int superb );

        int sgesvj(int matriint, char joba, char jobu, char jobv,int m, int n, long a, int lda,
                          long sva, int mv, long v, int ldv,
                          long stat );
        int dgesvj(int matriint, char joba, char jobu, char jobv,int m, int n, long a,int lda, long sva, int mv,
                          long v, int ldv, long stat );
        int cgesvj(int matriint, char joba, char jobu, char jobv,int m, int n, IComplexNumber a,int lda, long sva, int mv,
                          IComplexNumber v, int ldv, long stat );
        int zgesvj(int matriint, char joba, char jobu, char jobv,int m, int n, IComplexNumber a,int lda, long sva, int mv,
                          IComplexNumber v, int ldv, long stat );

        int sgesvx(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r, long c,
                          long b, int ldb, long x, int ldx,
                          long rcond, long ferr, long berr,
                          long rpivot );
        int dgesvx(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r, long c,
                          long b, int ldb, long x, int ldx,
                          long rcond, long ferr, long berr,
                          long rpivot );
        int cgesvx(int matriint, char fact, char trans,int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr,
                          long rpivot );
        int zgesvx(int matriint, char fact, char trans,int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr,
                          long rpivot );

        int sgesvxx(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r, long c,
                           long b, int ldb, long x, int ldx,
                           long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );
        int dgesvxx(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r, long c,
                           long b, int ldb, long x,int ldx, long rcond, long rpvgrw,
                           long berr, int int_bnds,
                           long erint_norm, long erint_comp,int nparams, long params );
        int cgesvxx(int matriint, char fact, char trans,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                           IComplexNumber b, int ldb,
                           IComplexNumber x, int ldx,
                           long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );
        int zgesvxx(int matriint, char fact, char trans,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                           IComplexNumber b, int ldb,
                           IComplexNumber x, int ldx,
                           long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );

        int sgetf2(int matriint, int m, int n,
                          long a, int lda, int ipiv );
        int dgetf2(int matriint, int m, int n,
                          long a, int lda, int ipiv );
        int cgetf2(int matriint, int m, int n,
                          IComplexNumber a, int lda,int ipiv );
        int zgetf2(int matriint, int m, int n,
                          IComplexNumber a, int lda,int ipiv );

        int sgetrf(int matriint, int m, int n,
                          long a, int lda, int ipiv );
        int dgetrf(int matriint, int m, int n,
                          long a, int lda, int ipiv );
        int cgetrf(int matriint, int m, int n,
                          IComplexNumber a, int lda,int ipiv );
        int zgetrf(int matriint, int m, int n,
                          IComplexNumber a, int lda,int ipiv );

        int sgetrf2(int matriint, int m, int n,
                           long a, int lda, int ipiv );
        int dgetrf2(int matriint, int m, int n,
                           long a, int lda, int ipiv );
        int cgetrf2(int matriint, int m, int n,
                           IComplexNumber a, int lda,int ipiv );
        int zgetrf2(int matriint, int m, int n,
                           IComplexNumber a, int lda,int ipiv );

        int sgetri(int matriint, int n, long a,int lda,int ipiv );
        int dgetri(int matriint, int n, long a,int lda,int ipiv );
        int cgetri(int matriint, int n,
                          IComplexNumber a, int lda,int ipiv );
        int zgetri(int matriint, int n,
                          IComplexNumber a, int lda,int ipiv );

        int sgetrs(int matriint, char trans, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
        int dgetrs(int matriint, char trans, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
        int cgetrs(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                          IComplexNumber b, int ldb );
        int zgetrs(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                          IComplexNumber b, int ldb );

        int sggbak(int matriint, char job, char side, int n,int ilo, int ihi,  long lscale,
                          long rscale, int m, long v,int ldv );
        int dggbak(int matriint, char job, char side, int n,int ilo, int ihi,  long lscale,
                          long rscale, int m, long v,int ldv );
        int cggbak(int matriint, char job, char side, int n,int ilo, int ihi,  long lscale,
                          long rscale, int m,
                          IComplexNumber v, int ldv );
        int zggbak(int matriint, char job, char side, int n,int ilo, int ihi,  long lscale,
                          long rscale, int m,
                          IComplexNumber v, int ldv );

        int sggbal(int matriint, char job, int n, long a,int lda, long b, int ldb,int ilo, int ihi, long lscale,
                          long rscale );
        int dggbal(int matriint, char job, int n, long a,int lda, long b, int ldb,int ilo, int ihi, long lscale,
                          long rscale );
        int cggbal(int matriint, char job, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,int ilo, int ihi, long lscale,
                          long rscale );
        int zggbal(int matriint, char job, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,int ilo, int ihi, long lscale,
                          long rscale );

        int sgges(int matriint, char jobvsl, char jobvsr, char sort,int selctg, int n, long a,int lda, long b, int ldb,int sdim, long alphar, long alphai,
                         long beta, long vsl, int ldvsl, long vsr,int ldvsr );
        int dgges(int matriint, char jobvsl, char jobvsr, char sort,int selctg, int n, long a,int lda, long b, int ldb,int sdim, long alphar, long alphai,
                         long beta, long vsl, int ldvsl,
                         long vsr, int ldvsr );
        int cgges(int matriint, char jobvsl, char jobvsr, char sort,int selctg, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                         IComplexNumber beta, IComplexNumber vsl,int ldvsl, IComplexNumber vsr,int ldvsr );
        int zgges(int matriint, char jobvsl, char jobvsr, char sort,int selctg, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                         IComplexNumber beta,
                         IComplexNumber vsl, int ldvsl,
                         IComplexNumber vsr, int ldvsr );

        int sgges3(int matriint, char jobvsl, char jobvsr,
                          char sort, int selctg, int n,
                          long a, int lda, long b, int ldb,int sdim, long alphar, long alphai,
                          long beta, long vsl, int ldvsl,
                          long vsr, int ldvsr );
        int dgges3(int matriint, char jobvsl, char jobvsr,
                          char sort, int selctg, int n,
                          long a, int lda, long b, int ldb,int sdim, long alphar, long alphai,
                          long beta, long vsl, int ldvsl,
                          long vsr, int ldvsr );
        int cgges3(int matriint, char jobvsl, char jobvsr,
                          char sort, int selctg, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                          IComplexNumber beta,
                          IComplexNumber vsl, int ldvsl,
                          IComplexNumber vsr, int ldvsr );
        int zgges3(int matriint, char jobvsl, char jobvsr,
                          char sort, int selctg, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                          IComplexNumber beta,
                          IComplexNumber vsl, int ldvsl,
                          IComplexNumber vsr, int ldvsr );

        int sggesx(int matriint, char jobvsl, char jobvsr,
                          char sort, int selctg, char sense,int n, long a, int lda, long b,int ldb, int sdim, long alphar,
                          long alphai, long beta, long vsl,int ldvsl, long vsr, int ldvsr,
                          long rconde, long rcondv );
        int dggesx(int matriint, char jobvsl, char jobvsr,
                          char sort, int selctg, char sense,int n, long a, int lda, long b,int ldb, int sdim, long alphar,
                          long alphai, long beta, long vsl,int ldvsl, long vsr, int ldvsr,
                          long rconde, long rcondv );
        int cggesx(int matriint, char jobvsl, char jobvsr,
                          char sort, int selctg, char sense,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, int sdim,
                          IComplexNumber alpha,
                          IComplexNumber beta,
                          IComplexNumber vsl, int ldvsl,
                          IComplexNumber vsr, int ldvsr,
                          long rconde, long rcondv );
        int zggesx(int matriint, char jobvsl, char jobvsr,
                          char sort, int selctg, char sense,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, int sdim,
                          IComplexNumber alpha,
                          IComplexNumber beta,
                          IComplexNumber vsl, int ldvsl,
                          IComplexNumber vsr, int ldvsr,
                          long rconde, long rcondv );

        int sggev(int matriint, char jobvl, char jobvr,int n, long a, int lda, long b,int ldb, long alphar, long alphai,
                         long beta, long vl, int ldvl, long vr,int ldvr );
        int dggev(int matriint, char jobvl, char jobvr,int n, long a, int lda, long b,int ldb, long alphar, long alphai,
                         long beta, long vl, int ldvl, long vr,int ldvr );
        int cggev(int matriint, char jobvl, char jobvr,int n, IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber alpha,
                         IComplexNumber beta, IComplexNumber vl,int ldvl, IComplexNumber vr,int ldvr );
        int zggev(int matriint, char jobvl, char jobvr,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber alpha,
                         IComplexNumber beta,
                         IComplexNumber vl, int ldvl,
                         IComplexNumber vr, int ldvr );

        int sggev3(int matriint, char jobvl, char jobvr,int n, long a, int lda,
                          long b, int ldb,
                          long alphar, long alphai, long beta,
                          long vl, int ldvl,
                          long vr, int ldvr );
        int dggev3(int matriint, char jobvl, char jobvr,int n, long a, int lda,
                          long b, int ldb,
                          long alphar, long alphai, long beta,
                          long vl, int ldvl,
                          long vr, int ldvr );
        int cggev3(int matriint, char jobvl, char jobvr,int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber alpha,
                          IComplexNumber beta,
                          IComplexNumber vl, int ldvl,
                          IComplexNumber vr, int ldvr );
        int zggev3(int matriint, char jobvl, char jobvr,int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber alpha,
                          IComplexNumber beta,
                          IComplexNumber vl, int ldvl,
                          IComplexNumber vr, int ldvr );

        int sggevx(int matriint, char balanc, char jobvl,
                          char jobvr, char sense, int n, long a,int lda, long b, int ldb,
                          long alphar, long alphai, long beta, long vl,int ldvl, long vr, int ldvr,int ilo, int ihi, long lscale,
                          long rscale, long abnrm, long bbnrm,
                          long rconde, long rcondv );
        int dggevx(int matriint, char balanc, char jobvl,
                          char jobvr, char sense, int n, long a,int lda, long b, int ldb,
                          long alphar, long alphai, long beta,
                          long vl, int ldvl, long vr,int ldvr, int ilo, int ihi,
                          long lscale, long rscale, long abnrm,
                          long bbnrm, long rconde, long rcondv );
        int cggevx(int matriint, char balanc, char jobvl,
                          char jobvr, char sense, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber alpha,
                          IComplexNumber beta, IComplexNumber vl,int ldvl, IComplexNumber vr,int ldvr, int ilo, int ihi,
                          long lscale, long rscale, long abnrm,
                          long bbnrm, long rconde, long rcondv );
        int zggevx(int matriint, char balanc, char jobvl,
                          char jobvr, char sense, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber alpha,
                          IComplexNumber beta,
                          IComplexNumber vl, int ldvl,
                          IComplexNumber vr, int ldvr,int  ilo, int ihi, long lscale,
                          long rscale, long abnrm, long bbnrm,
                          long rconde, long rcondv );

        int sggglm(int matriint, int n, int m,int p, long a, int lda, long b,int ldb, long d, long x, long y );
        int dggglm(int matriint, int n, int m,int p, long a, int lda, long b,int ldb, long d, long x, long y );
        int cggglm(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber d,
                          IComplexNumber x, IComplexNumber y );
        int zggglm(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber d,
                          IComplexNumber x, IComplexNumber y );

        int sgghrd(int matriint, char compq, char compz,int n, int ilo, int ihi,
                          long a, int lda, long b, int ldb,
                          long q, int ldq, long z, int ldz );
        int dgghrd(int matriint, char compq, char compz,int n, int ilo, int ihi,
                          long a, int lda, long b, int ldb,
                          long q, int ldq, long z,int ldz );
        int cgghrd(int matriint, char compq, char compz,int n, int ilo, int ihi,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber q, int ldq,
                          IComplexNumber z, int ldz );
        int zgghrd(int matriint, char compq, char compz,int n, int ilo, int ihi,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber q, int ldq,
                          IComplexNumber z, int ldz );

        int sgghd3(int matriint, char compq, char compz,int n, int ilo, int ihi,
                          long a, int lda, long b, int ldb,
                          long q, int ldq, long z, int ldz );
        int dgghd3(int matriint, char compq, char compz,int n, int ilo, int ihi,
                          long a, int lda, long b, int ldb,
                          long q, int ldq, long z,int ldz );
        int cgghd3(int matriint, char compq, char compz,int n, int ilo, int ihi,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber q, int ldq,
                          IComplexNumber z, int ldz );
        int zgghd3(int matriint, char compq, char compz,int n, int ilo, int ihi,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber q, int ldq,
                          IComplexNumber z, int ldz );

        int sgglse(int matriint, int m, int n,int p, long a, int lda, long b,int ldb, long c, long d, long x );
        int dgglse(int matriint, int m, int n,int p, long a, int lda, long b,int ldb, long c, long d, long x );
        int cgglse(int matriint, int m, int n,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber c,
                          IComplexNumber d, IComplexNumber x );
        int zgglse(int matriint, int m, int n,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber c,
                          IComplexNumber d, IComplexNumber x );

        int sggqrf(int matriint, int n, int m,int p, long a, int lda, long taua,
                          long b, int ldb, long taub );
        int dggqrf(int matriint, int n, int m,int p, long a, int lda,
                          long taua, long b, int ldb,
                          long taub );
        int cggqrf(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber taua,
                          IComplexNumber b, int ldb,
                          IComplexNumber taub );
        int zggqrf(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber taua,
                          IComplexNumber b, int ldb,
                          IComplexNumber taub );

        int sggrqf(int matriint, int m, int p,int n, long a, int lda, long taua,
                          long b, int ldb, long taub );
        int dggrqf(int matriint, int m, int p,int n, long a, int lda,
                          long taua, long b, int ldb,
                          long taub );
        int cggrqf(int matriint, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber taua,
                          IComplexNumber b, int ldb,
                          IComplexNumber taub );
        int zggrqf(int matriint, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber taua,
                          IComplexNumber b, int ldb,
                          IComplexNumber taub );

        int sggsvd(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l, long a,int lda, long b, int ldb,
                          long alpha, long beta, long u, int ldu,
                          long v, int ldv, long q, int ldq,int iwork );
        int dggsvd(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l, long a,int lda, long b, int ldb,
                          long alpha, long beta, long u,int ldu, long v, int ldv, long q,int ldq, int iwork );
        int cggsvd(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          long alpha, long beta, IComplexNumber u,int ldu, IComplexNumber v,int ldv, IComplexNumber q,int ldq, int iwork );
        int zggsvd(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          long alpha, long beta,
                          IComplexNumber u, int ldu,
                          IComplexNumber v, int ldv,
                          IComplexNumber q, int ldq,int iwork );

        int sggsvd3(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l, long a,int lda, long b, int ldb,
                           long alpha, long beta, long u, int ldu,
                           long v, int ldv, long q, int ldq,int iwork );
        int dggsvd3(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l, long a,int lda, long b, int ldb,
                           long alpha, long beta, long u,int ldu, long v, int ldv, long q,int ldq, int iwork );
        int cggsvd3(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,
                           long alpha, long beta, IComplexNumber u,int ldu, IComplexNumber v,int ldv, IComplexNumber q,int ldq, int iwork );
        int zggsvd3(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,
                           long alpha, long beta,
                           IComplexNumber u, int ldu,
                           IComplexNumber v, int ldv,
                           IComplexNumber q, int ldq,int iwork );

        int sggsvp(int matriint, char jobu, char jobv, char jobq,int m, int p, int n, long a,int lda, long b, int ldb, float tola,
                          float tolb, int k, int l, long u,int ldu, long v, int ldv, long q,int ldq );
        int dggsvp(int matriint, char jobu, char jobv, char jobq,int m, int p, int n, long a,int lda, long b, int ldb,
                          double tola, double tolb, int k,int l, long u, int ldu, long v,int ldv, long q, int ldq );
        int cggsvp(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb, float tola,
                          float tolb, int k, int l,
                          IComplexNumber u, int ldu,
                          IComplexNumber v, int ldv,
                          IComplexNumber q, int ldq );
        int zggsvp(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          double tola, double tolb, int k,int l, IComplexNumber u,int ldu, IComplexNumber v,int ldv, IComplexNumber q,int ldq );

        int sggsvp3(int matriint, char jobu, char jobv, char jobq,int m, int p, int n, long a,int lda, long b, int ldb, float tola,
                           float tolb, int k, int l, long u,int ldu, long v, int ldv, long q,int ldq );
        int dggsvp3(int matriint, char jobu, char jobv, char jobq,int m, int p, int n, long a,int lda, long b, int ldb,
                           double tola, double tolb, int k,int l, long u, int ldu, long v,int ldv, long q, int ldq );
        int cggsvp3(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb, float tola,
                           float tolb, int k, int l,
                           IComplexNumber u, int ldu,
                           IComplexNumber v, int ldv,
                           IComplexNumber q, int ldq );
        int zggsvp3(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,
                           double tola, double tolb, int k,int l, IComplexNumber u,int ldu, IComplexNumber v,int ldv, IComplexNumber q,int ldq );

        int sgtcon( char norm, int n,  long dl,
                           long d,  long du,  long du2,int ipiv, float anorm, long rcond );
        int dgtcon( char norm, int n,  long dl,
                           long d,  long du,  long du2,int ipiv, double anorm,
                           long rcond );
        int cgtcon( char norm, int n,
                           IComplexNumber dl,
                           IComplexNumber d,
                           IComplexNumber du,
                           IComplexNumber du2,int ipiv, float anorm, long rcond );
        int zgtcon( char norm, int n,
                           IComplexNumber dl,
                           IComplexNumber d,
                           IComplexNumber du,
                           IComplexNumber du2,int ipiv, double anorm,
                           long rcond );

        int sgtrfs(int matriint, char trans, int n,int nrhs,  long dl,  long d,
                          long du,  long dlf,  long df,
                          long duf,  long du2,int ipiv,  long b,int ldb, long x, int ldx,
                          long ferr, long berr );
        int dgtrfs(int matriint, char trans, int n,int nrhs,  long dl,  long d,
                          long du,  long dlf,
                          long df,  long duf,
                          long du2,int ipiv,
                          long b, int ldb, long x,int ldx, long ferr, long berr );
        int cgtrfs(int matriint, char trans, int n,int nrhs,  IComplexNumber dl,
                          IComplexNumber d,
                          IComplexNumber du,
                          IComplexNumber dlf,
                          IComplexNumber df,
                          IComplexNumber duf,
                          IComplexNumber du2,int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx, long ferr,
                          long berr );
        int zgtrfs(int matriint, char trans, int n,int nrhs,  IComplexNumber dl,
                          IComplexNumber d,
                          IComplexNumber du,
                          IComplexNumber dlf,
                          IComplexNumber df,
                          IComplexNumber duf,
                          IComplexNumber du2,int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );

        int sgtsv(int matriint, int n, int nrhs,
                         long dl, long d, long du, long b,int ldb );
        int dgtsv(int matriint, int n, int nrhs,
                         long dl, long d, long du, long b,int ldb );
        int cgtsv(int matriint, int n, int nrhs,
                         IComplexNumber dl, IComplexNumber d,
                         IComplexNumber du, IComplexNumber b,int ldb );
        int zgtsv(int matriint, int n, int nrhs,
                         IComplexNumber dl, IComplexNumber d,
                         IComplexNumber du, IComplexNumber b,int ldb );

        int sgtsvx(int matriint, char fact, char trans,int n, int nrhs,  long dl,
                          long d,  long du, long dlf,
                          long df, long duf, long du2, int ipiv,
                          long b, int ldb, long x,int ldx, long rcond, long ferr,
                          long berr );
        int dgtsvx(int matriint, char fact, char trans,int n, int nrhs,  long dl,
                          long d,  long du, long dlf,
                          long df, long duf, long du2,int ipiv,  long b, int ldb,
                          long x, int ldx, long rcond,
                          long ferr, long berr );
        int cgtsvx(int matriint, char fact, char trans,int n, int nrhs,
                          IComplexNumber dl,
                          IComplexNumber d,
                          IComplexNumber du,
                          IComplexNumber dlf, IComplexNumber df,
                          IComplexNumber duf, IComplexNumber du2,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                          long berr );
        int zgtsvx(int matriint, char fact, char trans,int n, int nrhs,
                          IComplexNumber dl,
                          IComplexNumber d,
                          IComplexNumber du,
                          IComplexNumber dlf,
                          IComplexNumber df,
                          IComplexNumber duf,
                          IComplexNumber du2, int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr );

        int sgttrf(int n, long dl, long d, long du,
                          long du2, int ipiv );
        int dgttrf(int n, long dl, long d, long du,
                          long du2, int ipiv );
        int cgttrf(int n, IComplexNumber dl,
                          IComplexNumber d, IComplexNumber du,
                          IComplexNumber du2, int ipiv );
        int zgttrf(int n, IComplexNumber dl,
                          IComplexNumber d, IComplexNumber du,
                          IComplexNumber du2, int ipiv );

        int sgttrs(int matriint, char trans, int n,int nrhs,  long dl,  long d,
                          long du,  long du2,int ipiv, long b, int ldb );
        int dgttrs(int matriint, char trans, int n,int nrhs,  long dl,  long d,
                          long du,  long du2,int ipiv, long b, int ldb );
        int cgttrs(int matriint, char trans, int n,int nrhs,  IComplexNumber dl,
                          IComplexNumber d,
                          IComplexNumber du,
                          IComplexNumber du2,int ipiv, IComplexNumber b,int ldb );
        int zgttrs(int matriint, char trans, int n,int nrhs,  IComplexNumber dl,
                          IComplexNumber d,
                          IComplexNumber du,
                          IComplexNumber du2,int ipiv, IComplexNumber b,int ldb );

        int chbev(int matriint, char jobz, char uplo, int n,int kd, IComplexNumber ab,int ldab, long w, IComplexNumber z,int ldz );
        int zhbev(int matriint, char jobz, char uplo, int n,int kd, IComplexNumber ab,int ldab, long w, IComplexNumber z,int ldz );

        int chbevd(int matriint, char jobz, char uplo, int n,int kd, IComplexNumber ab,int ldab, long w, IComplexNumber z,int ldz );
        int zhbevd(int matriint, char jobz, char uplo, int n,int kd, IComplexNumber ab,int ldab, long w, IComplexNumber z,int ldz );

        int chbevx(int matriint, char jobz, char range, char uplo,int n, int kd,
                          IComplexNumber ab, int ldab,
                          IComplexNumber q, int ldq, float vl,
                          float vu, int il, int iu, float abstol,int m, long w, IComplexNumber z,int ldz, int ifail );
        int zhbevx(int matriint, char jobz, char range, char uplo,int n, int kd,
                          IComplexNumber ab, int ldab,
                          IComplexNumber q, int ldq, double vl,
                          double vu, int il, int iu,
                          double abstol, int m, long w,
                          IComplexNumber z, int ldz,int ifail );

        int chbgst(int matriint, char vect, char uplo, int n,int ka, int kb,
                          IComplexNumber ab, int ldab,
                          IComplexNumber bb, int ldbb,
                          IComplexNumber x, int ldx );
        int zhbgst(int matriint, char vect, char uplo, int n,int ka, int kb,
                          IComplexNumber ab, int ldab,
                          IComplexNumber bb, int ldbb,
                          IComplexNumber x, int ldx );

        int chbgv(int matriint, char jobz, char uplo, int n,int ka, int kb,
                         IComplexNumber ab, int ldab,
                         IComplexNumber bb, int ldbb, long w,
                         IComplexNumber z, int ldz );
        int zhbgv(int matriint, char jobz, char uplo, int n,int ka, int kb,
                         IComplexNumber ab, int ldab,
                         IComplexNumber bb, int ldbb, long w,
                         IComplexNumber z, int ldz );

        int chbgvd(int matriint, char jobz, char uplo, int n,int ka, int kb,
                          IComplexNumber ab, int ldab,
                          IComplexNumber bb, int ldbb, long w,
                          IComplexNumber z, int ldz );
        int zhbgvd(int matriint, char jobz, char uplo, int n,int ka, int kb,
                          IComplexNumber ab, int ldab,
                          IComplexNumber bb, int ldbb,
                          long w, IComplexNumber z,int ldz );

        int chbgvx(int matriint, char jobz, char range, char uplo,int n, int ka, int kb,
                          IComplexNumber ab, int ldab,
                          IComplexNumber bb, int ldbb,
                          IComplexNumber q, int ldq, float vl,
                          float vu, int il, int iu, float abstol,int m, long w, IComplexNumber z,int ldz, int ifail );
        int zhbgvx(int matriint, char jobz, char range, char uplo,int n, int ka, int kb,
                          IComplexNumber ab, int ldab,
                          IComplexNumber bb, int ldbb,
                          IComplexNumber q, int ldq, double vl,
                          double vu, int il, int iu,
                          double abstol, int m, long w,
                          IComplexNumber z, int ldz,int ifail );

        int chbtrd(int matriint, char vect, char uplo, int n,int kd, IComplexNumber ab,int ldab, long d, long e,
                          IComplexNumber q, int ldq );
        int zhbtrd(int matriint, char vect, char uplo, int n,int kd, IComplexNumber ab,int ldab, long d, long e,
                          IComplexNumber q, int ldq );

        int checon(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv, float anorm, long rcond );
        int zhecon(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv, double anorm,
                          long rcond );

        int cheequb(int matriint, char uplo, int n,
                           IComplexNumber a, int lda,
                           long s, long scond, long amax );
        int zheequb(int matriint, char uplo, int n,
                           IComplexNumber a, int lda,
                           long s, long scond, long amax );

        int cheev(int matriint, char jobz, char uplo, int n,
                         IComplexNumber a, int lda, long w );
        int zheev(int matriint, char jobz, char uplo, int n,
                         IComplexNumber a, int lda, long w );

        int cheevd(int matriint, char jobz, char uplo, int n,
                          IComplexNumber a, int lda, long w );
        int zheevd(int matriint, char jobz, char uplo, int n,
                          IComplexNumber a, int lda,
                          long w );

        int cheevr(int matriint, char jobz, char range, char uplo,int n, IComplexNumber a,int lda, float vl, float vu, int il,int iu, float abstol, int m, long w,
                          IComplexNumber z, int ldz,int isuppz );
        int zheevr(int matriint, char jobz, char range, char uplo,int n, IComplexNumber a,int lda, double vl, double vu, int il,int iu, double abstol, int m,
                          long w, IComplexNumber z, int ldz,int isuppz );

        int cheevx(int matriint, char jobz, char range, char uplo,int n, IComplexNumber a,int lda, float vl, float vu, int il,int iu, float abstol, int m, long w,
                          IComplexNumber z, int ldz,int ifail );
        int zheevx(int matriint, char jobz, char range, char uplo,int n, IComplexNumber a,int lda, double vl, double vu, int il,int iu, double abstol, int m,
                          long w, IComplexNumber z, int ldz,int ifail );

        int chegst(int matriint, int itype, char uplo,int n, IComplexNumber a,int lda,  IComplexNumber b,int ldb );
        int zhegst(int matriint, int itype, char uplo,int n, IComplexNumber a,int lda,  IComplexNumber b,int ldb );

        int chegv(int matriint, int itype, char jobz,
                         char uplo, int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, long w );
        int zhegv(int matriint, int itype, char jobz,
                         char uplo, int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, long w );

        int chegvd(int matriint, int itype, char jobz,
                          char uplo, int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, long w );
        int zhegvd(int matriint, int itype, char jobz,
                          char uplo, int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, long w );

        int chegvx(int matriint, int itype, char jobz,
                          char range, char uplo, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb, float vl,
                          float vu, int il, int iu, float abstol,int m, long w, IComplexNumber z,int ldz, int ifail );
        int zhegvx(int matriint, int itype, char jobz,
                          char range, char uplo, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb, double vl,
                          double vu, int il, int iu,
                          double abstol, int m, long w,
                          IComplexNumber z, int ldz,int ifail );

        int cherfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx, long ferr,
                          long berr );
        int zherfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );

        int cherfsx(int matriint, char uplo, char equed,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,int ipiv,  long s,
                           IComplexNumber b, int ldb,
                           IComplexNumber x, int ldx,
                           long rcond, long berr, int int_bnds,
                           long erint_norm, long erint_comp,int nparams, long params );
        int zherfsx(int matriint, char uplo, char equed,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,int ipiv,  long s,
                           IComplexNumber b, int ldb,
                           IComplexNumber x, int ldx,
                           long rcond, long berr, int int_bnds,
                           long erint_norm, long erint_comp,int nparams, long params );

        int chesv(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                         IComplexNumber b, int ldb );
        int zhesv(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                         IComplexNumber b, int ldb );

        int chesvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber af,int ldaf, int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr );
        int zhesvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber af,int ldaf, int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr );

        int chesvxx(int matriint, char fact, char uplo,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                           IComplexNumber b, int ldb,
                           IComplexNumber x, int ldx,
                           long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );
        int zhesvxx(int matriint, char fact, char uplo,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                           IComplexNumber b, int ldb,
                           IComplexNumber x, int ldx,
                           long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );

        int chetrd(int matriint, char uplo, int n,
                          IComplexNumber a, int lda, long d,
                          long e, IComplexNumber tau );
        int zhetrd(int matriint, char uplo, int n,
                          IComplexNumber a, int lda, long d,
                          long e, IComplexNumber tau );

        int chetrf(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv );
        int zhetrf(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv );

        int chetri(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv );
        int zhetri(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv );

        int chetrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                          IComplexNumber b, int ldb );
        int zhetrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                          IComplexNumber b, int ldb );

        int chfrk(int matriint, char transr, char uplo, char trans,int n, int k, float alpha,
                         IComplexNumber a, int lda,
                         float beta, IComplexNumber c );
        int zhfrk(int matriint, char transr, char uplo, char trans,int n, int k, double alpha,
                         IComplexNumber a, int lda,
                         double beta, IComplexNumber c );

        int shgeqz(int matriint, char job, char compq, char compz,int n, int ilo, int ihi,
                          long h, int ldh, long t, int ldt,
                          long alphar, long alphai, long beta, long q,int ldq, long z, int ldz );
        int dhgeqz(int matriint, char job, char compq, char compz,int n, int ilo, int ihi,
                          long h, int ldh, long t, int ldt,
                          long alphar, long alphai, long beta,
                          long q, int ldq, long z,int ldz );
        int chgeqz(int matriint, char job, char compq, char compz,int n, int ilo, int ihi,
                          IComplexNumber h, int ldh,
                          IComplexNumber t, int ldt,
                          IComplexNumber alpha,
                          IComplexNumber beta, IComplexNumber q,int ldq, IComplexNumber z,int ldz );
        int zhgeqz(int matriint, char job, char compq, char compz,int n, int ilo, int ihi,
                          IComplexNumber h, int ldh,
                          IComplexNumber t, int ldt,
                          IComplexNumber alpha,
                          IComplexNumber beta,
                          IComplexNumber q, int ldq,
                          IComplexNumber z, int ldz );

        int chpcon(int matriint, char uplo, int n,
                          IComplexNumber ap,int ipiv, float anorm, long rcond );
        int zhpcon(int matriint, char uplo, int n,
                          IComplexNumber ap,int ipiv, double anorm,
                          long rcond );

        int chpev(int matriint, char jobz, char uplo, int n,
                         IComplexNumber ap, long w,
                         IComplexNumber z, int ldz );
        int zhpev(int matriint, char jobz, char uplo, int n,
                         IComplexNumber ap, long w,
                         IComplexNumber z, int ldz );

        int chpevd(int matriint, char jobz, char uplo, int n,
                          IComplexNumber ap, long w,
                          IComplexNumber z, int ldz );
        int zhpevd(int matriint, char jobz, char uplo, int n,
                          IComplexNumber ap, long w,
                          IComplexNumber z, int ldz );

        int chpevx(int matriint, char jobz, char range, char uplo,int n, IComplexNumber ap, float vl,
                          float vu, int il, int iu, float abstol,int m, long w, IComplexNumber z,int ldz, int ifail );
        int zhpevx(int matriint, char jobz, char range, char uplo,int n, IComplexNumber ap, double vl,
                          double vu, int il, int iu,
                          double abstol, int m, long w,
                          IComplexNumber z, int ldz,int ifail );

        int chpgst(int matriint, int itype, char uplo,int n, IComplexNumber ap,
                          IComplexNumber bp );
        int zhpgst(int matriint, int itype, char uplo,int n, IComplexNumber ap,
                          IComplexNumber bp );

        int chpgv(int matriint, int itype, char jobz,
                         char uplo, int n, IComplexNumber ap,
                         IComplexNumber bp, long w,
                         IComplexNumber z, int ldz );
        int zhpgv(int matriint, int itype, char jobz,
                         char uplo, int n, IComplexNumber ap,
                         IComplexNumber bp, long w,
                         IComplexNumber z, int ldz );

        int chpgvd(int matriint, int itype, char jobz,
                          char uplo, int n, IComplexNumber ap,
                          IComplexNumber bp, long w,
                          IComplexNumber z, int ldz );
        int zhpgvd(int matriint, int itype, char jobz,
                          char uplo, int n, IComplexNumber ap,
                          IComplexNumber bp, long w,
                          IComplexNumber z, int ldz );

        int chpgvx(int matriint, int itype, char jobz,
                          char range, char uplo, int n,
                          IComplexNumber ap, IComplexNumber bp,
                          float vl, float vu, int il, int iu,
                          float abstol, int m, long w,
                          IComplexNumber z, int ldz,int ifail );
        int zhpgvx(int matriint, int itype, char jobz,
                          char range, char uplo, int n,
                          IComplexNumber ap, IComplexNumber bp,
                          double vl, double vu, int il, int iu,
                          double abstol, int m, long w,
                          IComplexNumber z, int ldz,int ifail );

        int chprfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                          IComplexNumber afp,int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx, long ferr,
                          long berr );
        int zhprfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                          IComplexNumber afp,int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );

        int chpsv(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
        int zhpsv(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );

        int chpsvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber ap,
                          IComplexNumber afp, int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr );
        int zhpsvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber ap,
                          IComplexNumber afp, int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr );

        int chptrd(int matriint, char uplo, int n,
                          IComplexNumber ap, long d, long e,
                          IComplexNumber tau );
        int zhptrd(int matriint, char uplo, int n,
                          IComplexNumber ap, long d, long e,
                          IComplexNumber tau );

        int chptrf(int matriint, char uplo, int n,
                          IComplexNumber ap, int ipiv );
        int zhptrf(int matriint, char uplo, int n,
                          IComplexNumber ap, int ipiv );

        int chptri(int matriint, char uplo, int n,
                          IComplexNumber ap,int ipiv );
        int zhptri(int matriint, char uplo, int n,
                          IComplexNumber ap,int ipiv );

        int chptrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
        int zhptrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );

        int shsein(int matriint, char job, char eigsrc, char initv,int select, int n,  long h,int ldh, long wr,  long wi,
                          long vl, int ldvl, long vr,int ldvr, int mm, int m,int ifaill, int ifailr );
        int dhsein(int matriint, char job, char eigsrc, char initv,int select, int n,
                          long h, int ldh, long wr,
                          long wi, long vl, int ldvl,
                          long vr, int ldvr, int mm,int m, int ifaill,int ifailr );
        int chsein(int matriint, char job, char eigsrc, char initv,int select, int n,
                          IComplexNumber h, int ldh,
                          IComplexNumber w, IComplexNumber vl,int ldvl, IComplexNumber vr,int ldvr, int mm, int m,int ifaill, int ifailr );
        int zhsein(int matriint, char job, char eigsrc, char initv,int select, int n,
                          IComplexNumber h, int ldh,
                          IComplexNumber w, IComplexNumber vl,int ldvl, IComplexNumber vr,int ldvr, int mm, int m,int ifaill, int ifailr );

        int shseqr(int matriint, char job, char compz, int n,int ilo, int ihi, long h,int ldh, long wr, long wi, long z,int ldz );
        int dhseqr(int matriint, char job, char compz, int n,int ilo, int ihi, long h,int ldh, long wr, long wi, long z,int ldz );
        int chseqr(int matriint, char job, char compz, int n,int ilo, int ihi,
                          IComplexNumber h, int ldh,
                          IComplexNumber w, IComplexNumber z,int ldz );
        int zhseqr(int matriint, char job, char compz, int n,int ilo, int ihi,
                          IComplexNumber h, int ldh,
                          IComplexNumber w, IComplexNumber z,int ldz );


        int slacpy(int matriint, char uplo, int m,int n,  long a, int lda, long b,int ldb );
        int dlacpy(int matriint, char uplo, int m,int n,  long a, int lda, long b,int ldb );
        int clacpy(int matriint, char uplo, int m,int n,  IComplexNumber a,int lda, IComplexNumber b,int ldb );
        int zlacpy(int matriint, char uplo, int m,int n,  IComplexNumber a,int lda, IComplexNumber b,int ldb );

        int clacp2(int matriint, char uplo, int m,int n,  long a, int lda,
                          IComplexNumber b, int ldb );
        int zlacp2(int matriint, char uplo, int m,int n,  long a, int lda,
                          IComplexNumber b, int ldb );

        int zlag2c(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber sa, int ldsa );

        int slag2d(int matriint, int m, int n,
                          long sa, int ldsa, long a,int lda );

        int dlag2s(int matriint, int m, int n,
                          long a, int lda, long sa,int ldsa );

        int clag2z(int matriint, int m, int n,
                          IComplexNumber sa, int ldsa,
                          IComplexNumber a, int lda );

        int slagge(int matriint, int m, int n,int kl, int ku,  long d,
                          long a, int lda, int iseed );
        int dlagge(int matriint, int m, int n,int kl, int ku,  long d,
                          long a, int lda, int iseed );
        int clagge(int matriint, int m, int n,int kl, int ku,  long d,
                          IComplexNumber a, int lda,int iseed );
        int zlagge(int matriint, int m, int n,int kl, int ku,  long d,
                          IComplexNumber a, int lda,int iseed );


        int slarfb(int matriint, char side, char trans, char direct,
                          char storev, int m, int n,int k,  long v, int ldv,
                          long t, int ldt, long c,int ldc );
        int dlarfb(int matriint, char side, char trans, char direct,
                          char storev, int m, int n,int k,  long v, int ldv,
                          long t, int ldt, long c,int ldc );
        int clarfb(int matriint, char side, char trans, char direct,
                          char storev, int m, int n,int k,  IComplexNumber v,int ldv,  IComplexNumber t,int ldt, IComplexNumber c,int ldc );
        int zlarfb(int matriint, char side, char trans, char direct,
                          char storev, int m, int n,int k,  IComplexNumber v,int ldv,  IComplexNumber t,int ldt, IComplexNumber c,int ldc );



        int slarft(int matriint, char direct, char storev,int n, int k,  long v,int ldv,  long tau, long t,int ldt );
        int dlarft(int matriint, char direct, char storev,int n, int k,  long v,int ldv,  long tau, long t,int ldt );
        int clarft(int matriint, char direct, char storev,int n, int k,
                          IComplexNumber v, int ldv,
                          IComplexNumber tau,
                          IComplexNumber t, int ldt );
        int zlarft(int matriint, char direct, char storev,int n, int k,
                          IComplexNumber v, int ldv,
                          IComplexNumber tau,
                          IComplexNumber t, int ldt );

        int slarfx(int matriint, char side, int m,int n,  long v, float tau, long c,int ldc, long work );
        int dlarfx(int matriint, char side, int m,int n,  long v, double tau, long c,int ldc, long work );
        int clarfx(int matriint, char side, int m,int n,  IComplexNumber v,
                          IComplexNumber tau, IComplexNumber c,int ldc, IComplexNumber work );
        int zlarfx(int matriint, char side, int m,int n,  IComplexNumber v,
                          IComplexNumber tau, IComplexNumber c,int ldc, IComplexNumber work );



        int slascl(int matriint, char type, int kl,int ku, float cfrom, float cto,int m, int n, long a,int lda );
        int dlascl(int matriint, char type, int kl,int ku, double cfrom, double cto,int m, int n, long a,int lda );
        int clascl(int matriint, char type, int kl,int ku, float cfrom, float cto,int m, int n, IComplexNumber a,int lda );
        int zlascl(int matriint, char type, int kl,int ku, double cfrom, double cto,int m, int n, IComplexNumber a,int lda );

        int slaset(int matriint, char uplo, int m,int n, float alpha, float beta, long a,int lda );
        int dlaset(int matriint, char uplo, int m,int n, double alpha, double beta, long a,int lda );
        int claset(int matriint, char uplo, int m,int n, IComplexNumber alpha,
                          IComplexNumber beta, IComplexNumber a,int lda );
        int zlaset(int matriint, char uplo, int m,int n, IComplexNumber alpha,
                          IComplexNumber beta, IComplexNumber a,int lda );

        int slasrt( char id, int n, long d );
        int dlasrt( char id, int n, long d );

        int slaswp(int matriint, int n, long a,int lda, int k1, int k2,int ipiv, int incx );
        int dlaswp(int matriint, int n, long a,int lda, int k1, int k2,int ipiv, int incx );
        int claswp(int matriint, int n,
                          IComplexNumber a, int lda,int k1, int k2,int ipiv,int incx );
        int zlaswp(int matriint, int n,
                          IComplexNumber a, int lda,int k1, int k2,int ipiv,int incx );

        int slatms(int matriint, int m, int n,
                          char dist, int iseed, char sym, long d,int mode, float cond, float dmax,int kl, int ku, char pack, long a,int lda );
        int dlatms(int matriint, int m, int n,
                          char dist, int iseed, char sym, long d,int mode, double cond, double dmax,int kl, int ku, char pack, long a,int lda );
        int clatms(int matriint, int m, int n,
                          char dist, int iseed, char sym, long d,int mode, float cond, float dmax,int kl, int ku, char pack,
                          IComplexNumber a, int lda );
        int zlatms(int matriint, int m, int n,
                          char dist, int iseed, char sym, long d,int mode, double cond, double dmax,int kl, int ku, char pack,
                          IComplexNumber a, int lda );

        int slauum(int matriint, char uplo, int n, long a,int lda );
        int dlauum(int matriint, char uplo, int n, long a,int lda );
        int clauum(int matriint, char uplo, int n,
                          IComplexNumber a, int lda );
        int zlauum(int matriint, char uplo, int n,
                          IComplexNumber a, int lda );

        int sopgtr(int matriint, char uplo, int n,
                          long ap,  long tau, long q,int ldq );
        int dopgtr(int matriint, char uplo, int n,
                          long ap,  long tau, long q,int ldq );

        int sopmtr(int matriint, char side, char uplo, char trans,int m, int n,  long ap,
                          long tau, long c, int ldc );
        int dopmtr(int matriint, char side, char uplo, char trans,int m, int n,  long ap,
                          long tau, long c, int ldc );

        int sorgbr(int matriint, char vect, int m,int n, int k, long a, int lda,
                          long tau );
        int dorgbr(int matriint, char vect, int m,int n, int k, long a,int lda,  long tau );

        int sorghr(int matriint, int n, int ilo,int ihi, long a, int lda,
                          long tau );
        int dorghr(int matriint, int n, int ilo,int ihi, long a, int lda,
                          long tau );

        int sorglq(int matriint, int m, int n,int k, long a, int lda,
                          long tau );
        int dorglq(int matriint, int m, int n,int k, long a, int lda,
                          long tau );

        int sorgql(int matriint, int m, int n,int k, long a, int lda,
                          long tau );
        int dorgql(int matriint, int m, int n,int k, long a, int lda,
                          long tau );

        int sorgqr(int matriint, int m, int n,int k, long a, int lda,
                          long tau );
        int dorgqr(int matriint, int m, int n,int k, long a, int lda,
                          long tau );

        int sorgrq(int matriint, int m, int n,int k, long a, int lda,
                          long tau );
        int dorgrq(int matriint, int m, int n,int k, long a, int lda,
                          long tau );

        int sorgtr(int matriint, char uplo, int n, long a,int lda,  long tau );
        int dorgtr(int matriint, char uplo, int n, long a,int lda,  long tau );

        int sormbr(int matriint, char vect, char side, char trans,int m, int n, int k,
                          long a, int lda,  long tau,
                          long c, int ldc );
        int dormbr(int matriint, char vect, char side, char trans,int m, int n, int k,
                          long a, int lda,  long tau,
                          long c, int ldc );

        int sormhr(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  long a, int lda,
                          long tau, long c, int ldc );
        int dormhr(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  long a, int lda,
                          long tau, long c, int ldc );

        int sormlq(int matriint, char side, char trans,int m, int n, int k,
                          long a, int lda,  long tau,
                          long c, int ldc );
        int dormlq(int matriint, char side, char trans,int m, int n, int k,
                          long a, int lda,  long tau,
                          long c, int ldc );

        int sormql(int matriint, char side, char trans,int m, int n, int k,
                          long a, int lda,  long tau,
                          long c, int ldc );
        int dormql(int matriint, char side, char trans,int m, int n, int k,
                          long a, int lda,  long tau,
                          long c, int ldc );

        int sormqr(int matriint, char side, char trans,int m, int n, int k,
                          long a, int lda,  long tau,
                          long c, int ldc );
        int dormqr(int matriint, char side, char trans,int m, int n, int k,
                          long a, int lda,  long tau,
                          long c, int ldc );

        int sormrq(int matriint, char side, char trans,int m, int n, int k,
                          long a, int lda,  long tau,
                          long c, int ldc );
        int dormrq(int matriint, char side, char trans,int m, int n, int k,
                          long a, int lda,  long tau,
                          long c, int ldc );

        int sormrz(int matriint, char side, char trans,int m, int n, int k,int l,  long a, int lda,
                          long tau, long c, int ldc );
        int dormrz(int matriint, char side, char trans,int m, int n, int k,int l,  long a, int lda,
                          long tau, long c, int ldc );

        int sormtr(int matriint, char side, char uplo, char trans,int m, int n,  long a,int lda,  long tau, long c,int ldc );
        int dormtr(int matriint, char side, char uplo, char trans,int m, int n,  long a,int lda,  long tau, long c,int ldc );

        int spbcon(int matriint, char uplo, int n,int kd,  long ab, int ldab,
                          float anorm, long rcond );
        int dpbcon(int matriint, char uplo, int n,int kd,  long ab, int ldab,
                          double anorm, long rcond );
        int cpbcon(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, float anorm, long rcond );
        int zpbcon(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, double anorm, long rcond );

        int spbequ(int matriint, char uplo, int n,int kd,  long ab, int ldab,
                          long s, long scond, long amax );
        int dpbequ(int matriint, char uplo, int n,int kd,  long ab, int ldab,
                          long s, long scond, long amax );
        int cpbequ(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, long s, long scond,
                          long amax );
        int zpbequ(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, long s, long scond,
                          long amax );

        int spbrfs(int matriint, char uplo, int n,int kd, int nrhs,  long ab,int ldab,  long afb, int ldafb,
                          long b, int ldb, long x,int ldx, long ferr, long berr );
        int dpbrfs(int matriint, char uplo, int n,int kd, int nrhs,  long ab,int ldab,  long afb, int ldafb,
                          long b, int ldb, long x,int ldx, long ferr, long berr );
        int cpbrfs(int matriint, char uplo, int n,int kd, int nrhs,
                          IComplexNumber ab, int ldab,
                          IComplexNumber afb, int ldafb,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx, long ferr,
                          long berr );
        int zpbrfs(int matriint, char uplo, int n,int kd, int nrhs,
                          IComplexNumber ab, int ldab,
                          IComplexNumber afb, int ldafb,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );

        int spbstf(int matriint, char uplo, int n,int kb, long bb, int ldbb );
        int dpbstf(int matriint, char uplo, int n,int kb, long bb, int ldbb );
        int cpbstf(int matriint, char uplo, int n,int kb, IComplexNumber bb,int ldbb );
        int zpbstf(int matriint, char uplo, int n,int kb, IComplexNumber bb,int ldbb );

        int spbsv(int matriint, char uplo, int n,int kd, int nrhs, long ab,int ldab, long b, int ldb );
        int dpbsv(int matriint, char uplo, int n,int kd, int nrhs, long ab,int ldab, long b, int ldb );
        int cpbsv(int matriint, char uplo, int n,int kd, int nrhs,
                         IComplexNumber ab, int ldab,
                         IComplexNumber b, int ldb );
        int zpbsv(int matriint, char uplo, int n,int kd, int nrhs,
                         IComplexNumber ab, int ldab,
                         IComplexNumber b, int ldb );

        int spbsvx(int matriint, char fact, char uplo, int n,int kd, int nrhs, long ab,int ldab, long afb, int ldafb,
                          String equed, long s, long b, int ldb,
                          long x, int ldx, long rcond, long ferr,
                          long berr );
        int dpbsvx(int matriint, char fact, char uplo, int n,int kd, int nrhs, long ab,int ldab, long afb, int ldafb,
                          String equed, long s, long b, int ldb,
                          long x, int ldx, long rcond,
                          long ferr, long berr );
        int cpbsvx(int matriint, char fact, char uplo, int n,int kd, int nrhs,
                          IComplexNumber ab, int ldab,
                          IComplexNumber afb, int ldafb,
                          String equed, long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                          long berr );
        int zpbsvx(int matriint, char fact, char uplo, int n,int kd, int nrhs,
                          IComplexNumber ab, int ldab,
                          IComplexNumber afb, int ldafb,
                          String equed, long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                          long berr );

        int spbtrf(int matriint, char uplo, int n,int kd, long ab, int ldab );
        int dpbtrf(int matriint, char uplo, int n,int kd, long ab, int ldab );
        int cpbtrf(int matriint, char uplo, int n,int kd, IComplexNumber ab,int ldab );
        int zpbtrf(int matriint, char uplo, int n,int kd, IComplexNumber ab,int ldab );

        int spbtrs(int matriint, char uplo, int n,int kd, int nrhs,  long ab,int ldab, long b, int ldb );
        int dpbtrs(int matriint, char uplo, int n,int kd, int nrhs,  long ab,int ldab, long b, int ldb );
        int cpbtrs(int matriint, char uplo, int n,int kd, int nrhs,
                          IComplexNumber ab, int ldab,
                          IComplexNumber b, int ldb );
        int zpbtrs(int matriint, char uplo, int n,int kd, int nrhs,
                          IComplexNumber ab, int ldab,
                          IComplexNumber b, int ldb );

        int spftrf(int matriint, char transr, char uplo,int n, long a );
        int dpftrf(int matriint, char transr, char uplo,int n, long a );
        int cpftrf(int matriint, char transr, char uplo,int n, IComplexNumber a );
        int zpftrf(int matriint, char transr, char uplo,int n, IComplexNumber a );

        int spftri(int matriint, char transr, char uplo,int n, long a );
        int dpftri(int matriint, char transr, char uplo,int n, long a );
        int cpftri(int matriint, char transr, char uplo,int n, IComplexNumber a );
        int zpftri(int matriint, char transr, char uplo,int n, IComplexNumber a );

        int spftrs(int matriint, char transr, char uplo,int n, int nrhs,  long a,
                          long b, int ldb );
        int dpftrs(int matriint, char transr, char uplo,int n, int nrhs,  long a,
                          long b, int ldb );
        int cpftrs(int matriint, char transr, char uplo,int n, int nrhs,
                          IComplexNumber a,
                          IComplexNumber b, int ldb );
        int zpftrs(int matriint, char transr, char uplo,int n, int nrhs,
                          IComplexNumber a,
                          IComplexNumber b, int ldb );

        int spocon(int matriint, char uplo, int n,
                          long a, int lda, float anorm,
                          long rcond );
        int dpocon(int matriint, char uplo, int n,
                          long a, int lda, double anorm,
                          long rcond );
        int cpocon(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,
                          float anorm, long rcond );
        int zpocon(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,
                          double anorm, long rcond );

        int spoequ(int matriint, int n,  long a,int lda, long s, long scond,
                          long amax );
        int dpoequ(int matriint, int n,  long a,int lda, long s, long scond,
                          long amax );
        int cpoequ(int matriint, int n,
                          IComplexNumber a, int lda,
                          long s, long scond, long amax );
        int zpoequ(int matriint, int n,
                          IComplexNumber a, int lda,
                          long s, long scond, long amax );

        int spoequb(int matriint, int n,  long a,int lda, long s, long scond,
                           long amax );
        int dpoequb(int matriint, int n,  long a,int lda, long s, long scond,
                           long amax );
        int cpoequb(int matriint, int n,
                           IComplexNumber a, int lda,
                           long s, long scond, long amax );
        int zpoequb(int matriint, int n,
                           IComplexNumber a, int lda,
                           long s, long scond, long amax );

        int sporfs(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                          long af, int ldaf,  long b,int ldb, long x, int ldx,
                          long ferr, long berr );
        int dporfs(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                          long af, int ldaf,  long b,int ldb, long x, int ldx,
                          long ferr, long berr );
        int cporfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long ferr, long berr );
        int zporfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long ferr, long berr );

        int sporfsx(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af, int ldaf,
                           long s,  long b, int ldb,
                           long x, int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );
        int dporfsx(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af, int ldaf,
                           long s,  long b, int ldb,
                           long x, int ldx, long rcond,
                           long berr, int int_bnds,
                           long erint_norm, long erint_comp,int nparams, long params );
        int cporfsx(int matriint, char uplo, char equed,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,
                           long s,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );
        int zporfsx(int matriint, char uplo, char equed,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,
                           long s,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );

        int sposv(int matriint, char uplo, int n,int nrhs, long a, int lda, long b,int ldb );
        int dposv(int matriint, char uplo, int n,int nrhs, long a, int lda, long b,int ldb );
        int cposv(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb );
        int zposv(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb );
        int dsposv(int matriint, char uplo, int n,int nrhs, long a, int lda,
                          long b, int ldb, long x, int ldx,int iter );
        int zcposv(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber x,int ldx, int iter );

        int sposvx(int matriint, char fact, char uplo, int n,int nrhs, long a, int lda, long af,int ldaf, String equed, long s, long b,int ldb, long x, int ldx,
                          long rcond, long ferr, long berr );
        int dposvx(int matriint, char fact, char uplo, int n,int nrhs, long a, int lda,
                          long af, int ldaf, String equed, long s,
                          long b, int ldb, long x, int ldx,
                          long rcond, long ferr, long berr );
        int cposvx(int matriint, char fact, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber af,int ldaf, String equed, long s,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr );
        int zposvx(int matriint, char fact, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber af,int ldaf, String equed, long s,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr );

        int sposvxx(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,
                           String equed, long s, long b, int ldb,
                           long x, int ldx, long rcond,
                           long rpvgrw, long berr, int int_bnds,
                           long erint_norm, long erint_comp,int nparams, long params );
        int dposvxx(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,
                           String equed, long s, long b, int ldb,
                           long x, int ldx, long rcond,
                           long rpvgrw, long berr, int int_bnds,
                           long erint_norm, long erint_comp,int nparams, long params );
        int cposvxx(int matriint, char fact, char uplo,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,
                           String equed, long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long rpvgrw,
                           long berr, int int_bnds,
                           long erint_norm, long erint_comp,int nparams, long params );
        int zposvxx(int matriint, char fact, char uplo,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,
                           String equed, long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long rpvgrw,
                           long berr, int int_bnds,
                           long erint_norm, long erint_comp,int nparams, long params );

        int spotrf2(int matriint, char uplo, int n, long a,int lda );
        int dpotrf2(int matriint, char uplo, int n, long a,int lda );
        int cpotrf2(int matriint, char uplo, int n,
                           IComplexNumber a, int lda );
        int zpotrf2(int matriint, char uplo, int n,
                           IComplexNumber a, int lda );

        int spotrf(int matriint, char uplo, int n, long a,int lda );
        int dpotrf(int matriint, char uplo, int n, long a,int lda );
        int cpotrf(int matriint, char uplo, int n,
                          IComplexNumber a, int lda );
        int zpotrf(int matriint, char uplo, int n,
                          IComplexNumber a, int lda );

        int spotri(int matriint, char uplo, int n, long a,int lda );
        int dpotri(int matriint, char uplo, int n, long a,int lda );
        int cpotri(int matriint, char uplo, int n,
                          IComplexNumber a, int lda );
        int zpotri(int matriint, char uplo, int n,
                          IComplexNumber a, int lda );

        int spotrs(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                          long b, int ldb );
        int dpotrs(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                          long b, int ldb );
        int cpotrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber b,int ldb );
        int zpotrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber b,int ldb );

        int sppcon(int matriint, char uplo, int n,
                          long ap, float anorm, long rcond );
        int dppcon(int matriint, char uplo, int n,
                          long ap, double anorm, long rcond );
        int cppcon(int matriint, char uplo, int n,
                          IComplexNumber ap, float anorm,
                          long rcond );
        int zppcon(int matriint, char uplo, int n,
                          IComplexNumber ap, double anorm,
                          long rcond );

        int sppequ(int matriint, char uplo, int n,
                          long ap, long s, long scond,
                          long amax );
        int dppequ(int matriint, char uplo, int n,
                          long ap, long s, long scond,
                          long amax );
        int cppequ(int matriint, char uplo, int n,
                          IComplexNumber ap, long s,
                          long scond, long amax );
        int zppequ(int matriint, char uplo, int n,
                          IComplexNumber ap, long s,
                          long scond, long amax );

        int spprfs(int matriint, char uplo, int n,int nrhs,  long ap,  long afp,
                          long b, int ldb, long x,int ldx, long ferr, long berr );
        int dpprfs(int matriint, char uplo, int n,int nrhs,  long ap,  long afp,
                          long b, int ldb, long x,int ldx, long ferr, long berr );
        int cpprfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                          IComplexNumber afp,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx, long ferr,
                          long berr );
        int zpprfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                          IComplexNumber afp,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );

        int sppsv(int matriint, char uplo, int n,int nrhs, long ap, long b,int ldb );
        int dppsv(int matriint, char uplo, int n,int nrhs, long ap, long b,int ldb );
        int cppsv(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,
                         IComplexNumber b, int ldb );
        int zppsv(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,
                         IComplexNumber b, int ldb );

        int sppsvx(int matriint, char fact, char uplo, int n,int nrhs, long ap, long afp, String equed,
                          long s, long b, int ldb, long x,int ldx, long rcond, long ferr,
                          long berr );
        int dppsvx(int matriint, char fact, char uplo, int n,int nrhs, long ap, long afp,
                          String equed, long s, long b, int ldb,
                          long x, int ldx, long rcond,
                          long ferr, long berr );
        int cppsvx(int matriint, char fact, char uplo, int n,int nrhs, IComplexNumber ap,
                          IComplexNumber afp, String equed, long s,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr );
        int zppsvx(int matriint, char fact, char uplo, int n,int nrhs, IComplexNumber ap,
                          IComplexNumber afp, String equed, long s,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr );

        int spptrf(int matriint, char uplo, int n,
                          long ap );
        int dpptrf(int matriint, char uplo, int n,
                          long ap );
        int cpptrf(int matriint, char uplo, int n,
                          IComplexNumber ap );
        int zpptrf(int matriint, char uplo, int n,
                          IComplexNumber ap );

        int spptri(int matriint, char uplo, int n,
                          long ap );
        int dpptri(int matriint, char uplo, int n,
                          long ap );
        int cpptri(int matriint, char uplo, int n,
                          IComplexNumber ap );
        int zpptri(int matriint, char uplo, int n,
                          IComplexNumber ap );

        int spptrs(int matriint, char uplo, int n,int nrhs,  long ap, long b,int ldb );
        int dpptrs(int matriint, char uplo, int n,int nrhs,  long ap, long b,int ldb );
        int cpptrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                          IComplexNumber b, int ldb );
        int zpptrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                          IComplexNumber b, int ldb );

        int spstrf(int matriint, char uplo, int n, long a,int lda, int piv, int rank,
                          float tol );
        int dpstrf(int matriint, char uplo, int n, long a,int lda, int piv, int rank,
                          double tol );
        int cpstrf(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int piv, int rank, float tol );
        int zpstrf(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int piv, int rank, double tol );

        int sptcon(int n,  long d,  long e,
                          float anorm, long rcond );
        int dptcon(int n,  long d,  long e,
                          double anorm, long rcond );
        int cptcon(int n,  long d,
                          IComplexNumber e, float anorm,
                          long rcond );
        int zptcon(int n,  long d,
                          IComplexNumber e, double anorm,
                          long rcond );

        int spteqr(int matriint, char compz, int n, long d,
                          long e, long z, int ldz );
        int dpteqr(int matriint, char compz, int n,
                          long d, long e, long z, int ldz );
        int cpteqr(int matriint, char compz, int n, long d,
                          long e, IComplexNumber z, int ldz );
        int zpteqr(int matriint, char compz, int n,
                          long d, long e, IComplexNumber z,int ldz );

        int sptrfs(int matriint, int n, int nrhs,
                          long d,  long e,  long df,
                          long ef,  long b, int ldb,
                          long x, int ldx, long ferr, long berr );
        int dptrfs(int matriint, int n, int nrhs,
                          long d,  long e,  long df,
                          long ef,  long b, int ldb,
                          long x, int ldx, long ferr,
                          long berr );
        int cptrfs(int matriint, char uplo, int n,int nrhs,  long d,
                          IComplexNumber e,  long df,
                          IComplexNumber ef,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx, long ferr,
                          long berr );
        int zptrfs(int matriint, char uplo, int n,int nrhs,  long d,
                          IComplexNumber e,  long df,
                          IComplexNumber ef,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );

        int sptsv(int matriint, int n, int nrhs,
                         long d, long e, long b, int ldb );
        int dptsv(int matriint, int n, int nrhs,
                         long d, long e, long b, int ldb );
        int cptsv(int matriint, int n, int nrhs,
                         long d, IComplexNumber e,
                         IComplexNumber b, int ldb );
        int zptsv(int matriint, int n, int nrhs,
                         long d, IComplexNumber e,
                         IComplexNumber b, int ldb );

        int sptsvx(int matriint, char fact, int n,int nrhs,  long d,  long e,
                          long df, long ef,  long b, int ldb,
                          long x, int ldx, long rcond, long ferr,
                          long berr );
        int dptsvx(int matriint, char fact, int n,int nrhs,  long d,  long e,
                          long df, long ef,  long b,int ldb, long x, int ldx,
                          long rcond, long ferr, long berr );
        int cptsvx(int matriint, char fact, int n,int nrhs,  long d,
                          IComplexNumber e, long df,
                          IComplexNumber ef,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr );
        int zptsvx(int matriint, char fact, int n,int nrhs,  long d,
                          IComplexNumber e, long df,
                          IComplexNumber ef,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr );

        int spttrf(int n, long d, long e );
        int dpttrf(int n, long d, long e );
        int cpttrf(int n, long d, IComplexNumber e );
        int zpttrf(int n, long d, IComplexNumber e );

        int spttrs(int matriint, int n, int nrhs,
                          long d,  long e, long b,int ldb );
        int dpttrs(int matriint, int n, int nrhs,
                          long d,  long e, long b,int ldb );
        int cpttrs(int matriint, char uplo, int n,int nrhs,  long d,
                          IComplexNumber e,
                          IComplexNumber b, int ldb );
        int zpttrs(int matriint, char uplo, int n,int nrhs,  long d,
                          IComplexNumber e,
                          IComplexNumber b, int ldb );

        int ssbev(int matriint, char jobz, char uplo, int n,int kd, long ab, int ldab, long w,
                         long z, int ldz );
        int dsbev(int matriint, char jobz, char uplo, int n,int kd, long ab, int ldab, long w,
                         long z, int ldz );

        int ssbevd(int matriint, char jobz, char uplo, int n,int kd, long ab, int ldab, long w,
                          long z, int ldz );
        int dsbevd(int matriint, char jobz, char uplo, int n,int kd, long ab, int ldab,
                          long w, long z, int ldz );

        int ssbevx(int matriint, char jobz, char range, char uplo,int n, int kd, long ab,int ldab, long q, int ldq, float vl,
                          float vu, int il, int iu, float abstol,int m, long w, long z, int ldz,int ifail );
        int dsbevx(int matriint, char jobz, char range, char uplo,int n, int kd, long ab,int ldab, long q, int ldq,
                          double vl, double vu, int il, int iu,
                          double abstol, int m, long w, long z,int ldz, int ifail );

        int ssbgst(int matriint, char vect, char uplo, int n,int ka, int kb, long ab,int ldab,  long bb, int ldbb,
                          long x, int ldx );
        int dsbgst(int matriint, char vect, char uplo, int n,int ka, int kb, long ab,int ldab,  long bb, int ldbb,
                          long x, int ldx );

        int ssbgv(int matriint, char jobz, char uplo, int n,int ka, int kb, long ab,int ldab, long bb, int ldbb, long w,
                         long z, int ldz );
        int dsbgv(int matriint, char jobz, char uplo, int n,int ka, int kb, long ab,int ldab, long bb, int ldbb,
                         long w, long z, int ldz );

        int ssbgvd(int matriint, char jobz, char uplo, int n,int ka, int kb, long ab,int ldab, long bb, int ldbb,
                          long w, long z, int ldz );
        int dsbgvd(int matriint, char jobz, char uplo, int n,int ka, int kb, long ab,int ldab, long bb, int ldbb,
                          long w, long z, int ldz );

        int ssbgvx(int matriint, char jobz, char range, char uplo,int n, int ka, int kb,
                          long ab, int ldab, long bb,int ldbb, long q, int ldq, float vl,
                          float vu, int il, int iu, float abstol,int m, long w, long z, int ldz,int ifail );
        int dsbgvx(int matriint, char jobz, char range, char uplo,int n, int ka, int kb,
                          long ab, int ldab, long bb,int ldbb, long q, int ldq,
                          double vl, double vu, int il, int iu,
                          double abstol, int m, long w, long z,int ldz, int ifail );

        int ssbtrd(int matriint, char vect, char uplo, int n,int kd, long ab, int ldab, long d,
                          long e, long q, int ldq );
        int dsbtrd(int matriint, char vect, char uplo, int n,int kd, long ab, int ldab,
                          long d, long e, long q, int ldq );

        int ssfrk(int matriint, char transr, char uplo, char trans,int n, int k, float alpha,
                         long a, int lda, float beta,
                         long c );
        int dsfrk(int matriint, char transr, char uplo, char trans,int n, int k, double alpha,
                         long a, int lda, double beta,
                         long c );

        int sspcon(int matriint, char uplo, int n,
                          long ap,int ipiv, float anorm,
                          long rcond );
        int dspcon(int matriint, char uplo, int n,
                          long ap,int ipiv,
                          double anorm, long rcond );
        int cspcon(int matriint, char uplo, int n,
                          IComplexNumber ap,int ipiv, float anorm, long rcond );
        int zspcon(int matriint, char uplo, int n,
                          IComplexNumber ap,int ipiv, double anorm,
                          long rcond );

        int sspev(int matriint, char jobz, char uplo, int n,
                         long ap, long w, long z, int ldz );
        int dspev(int matriint, char jobz, char uplo, int n,
                         long ap, long w, long z, int ldz );

        int sspevd(int matriint, char jobz, char uplo, int n,
                          long ap, long w, long z, int ldz );
        int dspevd(int matriint, char jobz, char uplo, int n,
                          long ap, long w, long z, int ldz );

        int sspevx(int matriint, char jobz, char range, char uplo,int n, long ap, float vl, float vu,int il, int iu, float abstol,int m, long w, long z, int ldz,int ifail );
        int dspevx(int matriint, char jobz, char range, char uplo,int n, long ap, double vl, double vu,int il, int iu, double abstol,int m, long w, long z, int ldz,int ifail );

        int sspgst(int matriint, int itype, char uplo,int n, long ap,  long bp );
        int dspgst(int matriint, int itype, char uplo,int n, long ap,  long bp );

        int sspgv(int matriint, int itype, char jobz,
                         char uplo, int n, long ap, long bp,
                         long w, long z, int ldz );
        int dspgv(int matriint, int itype, char jobz,
                         char uplo, int n, long ap, long bp,
                         long w, long z, int ldz );

        int sspgvd(int matriint, int itype, char jobz,
                          char uplo, int n, long ap, long bp,
                          long w, long z, int ldz );
        int dspgvd(int matriint, int itype, char jobz,
                          char uplo, int n, long ap, long bp,
                          long w, long z, int ldz );

        int sspgvx(int matriint, int itype, char jobz,
                          char range, char uplo, int n, long ap,
                          long bp, float vl, float vu, int il,int iu, float abstol, int m, long w,
                          long z, int ldz, int ifail );
        int dspgvx(int matriint, int itype, char jobz,
                          char range, char uplo, int n, long ap,
                          long bp, double vl, double vu, int il,int iu, double abstol, int m,
                          long w, long z, int ldz,int ifail );

        int ssprfs(int matriint, char uplo, int n,int nrhs,  long ap,  long afp,int ipiv,  long b,int ldb, long x, int ldx,
                          long ferr, long berr );
        int dsprfs(int matriint, char uplo, int n,int nrhs,  long ap,  long afp,int ipiv,  long b,int ldb, long x, int ldx,
                          long ferr, long berr );
        int csprfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                          IComplexNumber afp,int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx, long ferr,
                          long berr );
        int zsprfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                          IComplexNumber afp,int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );

        int sspsv(int matriint, char uplo, int n,int nrhs, long ap, int ipiv,
                         long b, int ldb );
        int dspsv(int matriint, char uplo, int n,int nrhs, long ap, int ipiv,
                         long b, int ldb );
        int cspsv(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
        int zspsv(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );

        int sspsvx(int matriint, char fact, char uplo, int n,int nrhs,  long ap, long afp,int ipiv,  long b, int ldb,
                          long x, int ldx, long rcond, long ferr,
                          long berr );
        int dspsvx(int matriint, char fact, char uplo, int n,int nrhs,  long ap, long afp,int ipiv,  long b, int ldb,
                          long x, int ldx, long rcond,
                          long ferr, long berr );
        int cspsvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber ap,
                          IComplexNumber afp, int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr );
        int zspsvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber ap,
                          IComplexNumber afp, int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr );

        int ssptrd(int matriint, char uplo, int n, long ap,
                          long d, long e, long tau );
        int dsptrd(int matriint, char uplo, int n,
                          long ap, long d, long e, long tau );

        int ssptrf(int matriint, char uplo, int n, long ap,int ipiv );
        int dsptrf(int matriint, char uplo, int n,
                          long ap, int ipiv );
        int csptrf(int matriint, char uplo, int n,
                          IComplexNumber ap, int ipiv );
        int zsptrf(int matriint, char uplo, int n,
                          IComplexNumber ap, int ipiv );

        int ssptri(int matriint, char uplo, int n, long ap,int ipiv );
        int dsptri(int matriint, char uplo, int n,
                          long ap,int ipiv );
        int csptri(int matriint, char uplo, int n,
                          IComplexNumber ap,int ipiv );
        int zsptri(int matriint, char uplo, int n,
                          IComplexNumber ap,int ipiv );

        int ssptrs(int matriint, char uplo, int n,int nrhs,  long ap,int ipiv, long b, int ldb );
        int dsptrs(int matriint, char uplo, int n,int nrhs,  long ap,int ipiv, long b, int ldb );
        int csptrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
        int zsptrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );

        int sstebz( char range, char order, int n, float vl,
                           float vu, int il, int iu, float abstol,
                           long d,  long e, int m,int nsplit, long w, int iblock,int isplit );
        int dstebz( char range, char order, int n, double vl,
                           double vu, int il, int iu,
                           double abstol,  long d,  long e,int m, int nsplit, long w,int iblock, int isplit );

        int sstedc(int matriint, char compz, int n, long d,
                          long e, long z, int ldz );
        int dstedc(int matriint, char compz, int n,
                          long d, long e, long z, int ldz );
        int cstedc(int matriint, char compz, int n, long d,
                          long e, IComplexNumber z, int ldz );
        int zstedc(int matriint, char compz, int n,
                          long d, long e, IComplexNumber z,int ldz );

        int sstegr(int matriint, char jobz, char range,int n, long d, long e, float vl, float vu,int il, int iu, float abstol,int m, long w, long z, int ldz,int isuppz );
        int dstegr(int matriint, char jobz, char range,int n, long d, long e, double vl,
                          double vu, int il, int iu,
                          double abstol, int m, long w, long z,int ldz, int isuppz );
        int cstegr(int matriint, char jobz, char range,int n, long d, long e, float vl, float vu,int il, int iu, float abstol,int m, long w, IComplexNumber z,int ldz, int isuppz );
        int zstegr(int matriint, char jobz, char range,int n, long d, long e, double vl,
                          double vu, int il, int iu,
                          double abstol, int m, long w,
                          IComplexNumber z, int ldz,int isuppz );

        int sstein(int matriint, int n,  long d,
                          long e, int m,  long w,int iblock,int isplit,
                          long z, int ldz, int ifailv );
        int dstein(int matriint, int n,  long d,
                          long e, int m,  long w,int iblock,int isplit,
                          long z, int ldz, int ifailv );
        int cstein(int matriint, int n,  long d,
                          long e, int m,  long w,int iblock,int isplit,
                          IComplexNumber z, int ldz,int ifailv );
        int zstein(int matriint, int n,  long d,
                          long e, int m,  long w,int iblock,int isplit,
                          IComplexNumber z, int ldz,int ifailv );

        int sstemr(int matriint, char jobz, char range,int n, long d, long e, float vl, float vu,int il, int iu, int m,
                          long w, long z, int ldz, int nzc,int isuppz, int tryrac );
        int dstemr(int matriint, char jobz, char range,int n, long d, long e, double vl,
                          double vu, int il, int iu,int m, long w, long z, int ldz,int nzc, int isuppz,int tryrac );
        int cstemr(int matriint, char jobz, char range,int n, long d, long e, float vl, float vu,int il, int iu, int m,
                          long w, IComplexNumber z, int ldz,int nzc, int isuppz,int tryrac );
        int zstemr(int matriint, char jobz, char range,int n, long d, long e, double vl,
                          double vu, int il, int iu,int m, long w, IComplexNumber z,int ldz, int nzc, int isuppz,int tryrac );

        int ssteqr(int matriint, char compz, int n, long d,
                          long e, long z, int ldz );
        int dsteqr(int matriint, char compz, int n,
                          long d, long e, long z, int ldz );
        int csteqr(int matriint, char compz, int n, long d,
                          long e, IComplexNumber z, int ldz );
        int zsteqr(int matriint, char compz, int n,
                          long d, long e, IComplexNumber z,int ldz );

        int ssterf(int n, long d, long e );
        int dsterf(int n, long d, long e );

        int sstev(int matriint, char jobz, int n, long d,
                         long e, long z, int ldz );
        int dstev(int matriint, char jobz, int n, long d,
                         long e, long z, int ldz );

        int sstevd(int matriint, char jobz, int n, long d,
                          long e, long z, int ldz );
        int dstevd(int matriint, char jobz, int n, long d,
                          long e, long z, int ldz );

        int sstevr(int matriint, char jobz, char range,int n, long d, long e, float vl, float vu,int il, int iu, float abstol,int m, long w, long z, int ldz,int isuppz );
        int dstevr(int matriint, char jobz, char range,int n, long d, long e, double vl,
                          double vu, int il, int iu,
                          double abstol, int m, long w, long z,int ldz, int isuppz );

        int sstevx(int matriint, char jobz, char range,int n, long d, long e, float vl, float vu,int il, int iu, float abstol,int m, long w, long z, int ldz,int ifail );
        int dstevx(int matriint, char jobz, char range,int n, long d, long e, double vl,
                          double vu, int il, int iu,
                          double abstol, int m, long w, long z,int ldz, int ifail );

        int ssycon(int matriint, char uplo, int n,
                          long a, int lda,int ipiv, float anorm, long rcond );
        int dsycon(int matriint, char uplo, int n,
                          long a, int lda,int ipiv, double anorm,
                          long rcond );
        int csycon(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv, float anorm, long rcond );
        int zsycon(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv, double anorm,
                          long rcond );

        int ssyequb(int matriint, char uplo, int n,
                           long a, int lda, long s,
                           long scond, long amax );
        int dsyequb(int matriint, char uplo, int n,
                           long a, int lda, long s,
                           long scond, long amax );
        int csyequb(int matriint, char uplo, int n,
                           IComplexNumber a, int lda,
                           long s, long scond, long amax );
        int zsyequb(int matriint, char uplo, int n,
                           IComplexNumber a, int lda,
                           long s, long scond, long amax );

        int ssyev(int matriint, char jobz, char uplo, int n,
                         long a, int lda, long w );
        int dsyev(int matriint, char jobz, char uplo, int n,
                         long a, int lda, long w );

        int ssyevd(int matriint, char jobz, char uplo, int n,
                          long a, int lda, long w );
        int dsyevd(int matriint, char jobz, char uplo, int n,
                          long a, int lda, long w );

        int ssyevr(int matriint, char jobz, char range, char uplo,int n, long a, int lda, float vl,
                          float vu, int il, int iu, float abstol,int m, long w, long z, int ldz,int isuppz );
        int dsyevr(int matriint, char jobz, char range, char uplo,int n, long a, int lda, double vl,
                          double vu, int il, int iu,
                          double abstol, int m, long w, long z,int ldz, int isuppz );

        int ssyevx(int matriint, char jobz, char range, char uplo,int n, long a, int lda, float vl,
                          float vu, int il, int iu, float abstol,int m, long w, long z, int ldz,int ifail );
        int dsyevx(int matriint, char jobz, char range, char uplo,int n, long a, int lda, double vl,
                          double vu, int il, int iu,
                          double abstol, int m, long w, long z,int ldz, int ifail );

        int ssygst(int matriint, int itype, char uplo,int n, long a, int lda,
                          long b, int ldb );
        int dsygst(int matriint, int itype, char uplo,int n, long a, int lda,
                          long b, int ldb );

        int ssygv(int matriint, int itype, char jobz,
                         char uplo, int n, long a, int lda,
                         long b, int ldb, long w );
        int dsygv(int matriint, int itype, char jobz,
                         char uplo, int n, long a, int lda,
                         long b, int ldb, long w );

        int ssygvd(int matriint, int itype, char jobz,
                          char uplo, int n, long a, int lda,
                          long b, int ldb, long w );
        int dsygvd(int matriint, int itype, char jobz,
                          char uplo, int n, long a, int lda,
                          long b, int ldb, long w );

        int ssygvx(int matriint, int itype, char jobz,
                          char range, char uplo, int n, long a,int lda, long b, int ldb, float vl,
                          float vu, int il, int iu, float abstol,int m, long w, long z, int ldz,int ifail );
        int dsygvx(int matriint, int itype, char jobz,
                          char range, char uplo, int n, long a,int lda, long b, int ldb, double vl,
                          double vu, int il, int iu,
                          double abstol, int m, long w, long z,int ldz, int ifail );

        int ssyrfs(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                          long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                          long ferr, long berr );
        int dsyrfs(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                          long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                          long ferr, long berr );
        int csyrfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx, long ferr,
                          long berr );
        int zsyrfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );

        int ssyrfsx(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af, int ldaf,int ipiv,  long s,
                           long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );
        int dsyrfsx(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af, int ldaf,int ipiv,  long s,
                           long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );
        int csyrfsx(int matriint, char uplo, char equed,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,int ipiv,  long s,
                           IComplexNumber b, int ldb,
                           IComplexNumber x, int ldx,
                           long rcond, long berr, int int_bnds,
                           long erint_norm, long erint_comp,int nparams, long params );
        int zsyrfsx(int matriint, char uplo, char equed,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,int ipiv,  long s,
                           IComplexNumber b, int ldb,
                           IComplexNumber x, int ldx,
                           long rcond, long berr, int int_bnds,
                           long erint_norm, long erint_comp,int nparams, long params );

        int ssysv(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb );
        int dsysv(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb );
        int csysv(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                         IComplexNumber b, int ldb );
        int zsysv(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                         IComplexNumber b, int ldb );

        int ssysvx(int matriint, char fact, char uplo, int n,int nrhs,  long a, int lda,
                          long af, int ldaf, int ipiv,
                          long b, int ldb, long x,int ldx, long rcond, long ferr,
                          long berr );
        int dsysvx(int matriint, char fact, char uplo, int n,int nrhs,  long a, int lda,
                          long af, int ldaf, int ipiv,
                          long b, int ldb, long x,int ldx, long rcond, long ferr,
                          long berr );
        int csysvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber af,int ldaf, int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr );
        int zsysvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber af,int ldaf, int ipiv,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long rcond, long ferr, long berr );

        int ssysvxx(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long s, long b,int ldb, long x, int ldx,
                           long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );
        int dsysvxx(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long s, long b,int ldb, long x, int ldx,
                           long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );
        int csysvxx(int matriint, char fact, char uplo,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                           IComplexNumber b, int ldb,
                           IComplexNumber x, int ldx,
                           long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );
        int zsysvxx(int matriint, char fact, char uplo,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                           IComplexNumber b, int ldb,
                           IComplexNumber x, int ldx,
                           long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                           long erint_comp, int nparams,
                           long params );

        int ssytrd(int matriint, char uplo, int n, long a,int lda, long d, long e, long tau );
        int dsytrd(int matriint, char uplo, int n, long a,int lda, long d, long e, long tau );

        int ssytrf(int matriint, char uplo, int n, long a,int lda, int ipiv );
        int dsytrf(int matriint, char uplo, int n, long a,int lda, int ipiv );
        int csytrf(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv );
        int zsytrf(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv );

        int ssytri(int matriint, char uplo, int n, long a,int lda,int ipiv );
        int dsytri(int matriint, char uplo, int n, long a,int lda,int ipiv );
        int csytri(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv );
        int zsytri(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv );

        int ssytrs(int matriint, char uplo, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
        int dsytrs(int matriint, char uplo, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
        int csytrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                          IComplexNumber b, int ldb );
        int zsytrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                          IComplexNumber b, int ldb );

        int stbcon(int matriint, char norm, char uplo, char diag,int n, int kd,  long ab,int ldab, long rcond );
        int dtbcon(int matriint, char norm, char uplo, char diag,int n, int kd,  long ab,int ldab, long rcond );
        int ctbcon(int matriint, char norm, char uplo, char diag,int n, int kd,
                          IComplexNumber ab, int ldab,
                          long rcond );
        int ztbcon(int matriint, char norm, char uplo, char diag,int n, int kd,
                          IComplexNumber ab, int ldab,
                          long rcond );

        int stbrfs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                          long ab, int ldab,  long b,int ldb,  long x, int ldx,
                          long ferr, long berr );
        int dtbrfs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                          long ab, int ldab,  long b,int ldb,  long x, int ldx,
                          long ferr, long berr );
        int ctbrfs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                          IComplexNumber ab, int ldab,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );
        int ztbrfs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                          IComplexNumber ab, int ldab,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );

        int stbtrs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                          long ab, int ldab, long b,int ldb );
        int dtbtrs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                          long ab, int ldab, long b,int ldb );
        int ctbtrs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                          IComplexNumber ab, int ldab,
                          IComplexNumber b, int ldb );
        int ztbtrs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                          IComplexNumber ab, int ldab,
                          IComplexNumber b, int ldb );

        int stfsm(int matriint, char transr, char side, char uplo,
                         char trans, char diag, int m, int n,
                         float alpha,  long a, long b,int ldb );
        int dtfsm(int matriint, char transr, char side, char uplo,
                         char trans, char diag, int m, int n,
                         double alpha,  long a, long b,int ldb );
        int ctfsm(int matriint, char transr, char side, char uplo,
                         char trans, char diag, int m, int n,
                         IComplexNumber alpha,
                         IComplexNumber a,
                         IComplexNumber b, int ldb );
        int ztfsm(int matriint, char transr, char side, char uplo,
                         char trans, char diag, int m, int n,
                         IComplexNumber alpha,
                         IComplexNumber a,
                         IComplexNumber b, int ldb );

        int stftri(int matriint, char transr, char uplo, char diag,int n, long a );
        int dtftri(int matriint, char transr, char uplo, char diag,int n, long a );
        int ctftri(int matriint, char transr, char uplo, char diag,int n, IComplexNumber a );
        int ztftri(int matriint, char transr, char uplo, char diag,int n, IComplexNumber a );

        int stfttp(int matriint, char transr, char uplo,int n,  long arf, long ap );
        int dtfttp(int matriint, char transr, char uplo,int n,  long arf, long ap );
        int ctfttp(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                          IComplexNumber ap );
        int ztfttp(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                          IComplexNumber ap );

        int stfttr(int matriint, char transr, char uplo,int n,  long arf, long a,int lda );
        int dtfttr(int matriint, char transr, char uplo,int n,  long arf, long a,int lda );
        int ctfttr(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                          IComplexNumber a, int lda );
        int ztfttr(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                          IComplexNumber a, int lda );

        int stgevc(int matriint, char side, char howmny,int select, int n,
                          long s, int lds,  long p,int ldp, long vl, int ldvl,
                          long vr, int ldvr, int mm,int m );
        int dtgevc(int matriint, char side, char howmny,int select, int n,
                          long s, int lds,  long p,int ldp, long vl, int ldvl,
                          long vr, int ldvr, int mm,int m );
        int ctgevc(int matriint, char side, char howmny,int select, int n,
                          IComplexNumber s, int lds,
                          IComplexNumber p, int ldp,
                          IComplexNumber vl, int ldvl,
                          IComplexNumber vr, int ldvr,int mm, int m );
        int ztgevc(int matriint, char side, char howmny,int select, int n,
                          IComplexNumber s, int lds,
                          IComplexNumber p, int ldp,
                          IComplexNumber vl, int ldvl,
                          IComplexNumber vr, int ldvr,int mm, int m );

        int stgexc(int matriint, int wantq,int wantz, int n, long a,int lda, long b, int ldb, long q,int ldq, long z, int ldz,int ifst, int ilst );
        int dtgexc(int matriint, int wantq,int wantz, int n, long a,int lda, long b, int ldb, long q,int ldq, long z, int ldz,int ifst, int ilst );
        int ctgexc(int matriint, int wantq,int wantz, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber q, int ldq,
                          IComplexNumber z, int ldz,int ifst, int ilst );
        int ztgexc(int matriint, int wantq,int wantz, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber q, int ldq,
                          IComplexNumber z, int ldz,int ifst, int ilst );

        int stgsen(int matriint, int ijob,int wantq, int wantz,int select, int n, long a,int lda, long b, int ldb,
                          long alphar, long alphai, long beta, long q,int ldq, long z, int ldz,int m, long pl, long pr, long dif );
        int dtgsen(int matriint, int ijob,int wantq, int wantz,int select, int n,
                          long a, int lda, long b, int ldb,
                          long alphar, long alphai, long beta,
                          long q, int ldq, long z, int ldz,int m, long pl, long pr, long dif );
        int ctgsen(int matriint, int ijob,int wantq, int wantz,int select, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber alpha,
                          IComplexNumber beta, IComplexNumber q,int ldq, IComplexNumber z,int ldz, int m, long pl, long pr,
                          long dif );
        int ztgsen(int matriint, int ijob,int wantq, int wantz,int select, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber alpha,
                          IComplexNumber beta,
                          IComplexNumber q, int ldq,
                          IComplexNumber z, int ldz,int m, long pl, long pr, long dif );

        int stgsja(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,int k, int l, long a, int lda,
                          long b, int ldb, float tola, float tolb,
                          long alpha, long beta, long u, int ldu,
                          long v, int ldv, long q, int ldq,int ncycle );
        int dtgsja(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,int k, int l, long a,int lda, long b, int ldb,
                          double tola, double tolb, long alpha,
                          long beta, long u, int ldu, long v,int ldv, long q, int ldq,int ncycle );
        int ctgsja(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,int k, int l, IComplexNumber a,int lda, IComplexNumber b,int ldb, float tola, float tolb, long alpha,
                          long beta, IComplexNumber u, int ldu,
                          IComplexNumber v, int ldv,
                          IComplexNumber q, int ldq,int ncycle );
        int ztgsja(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,int k, int l, IComplexNumber a,int lda, IComplexNumber b,int ldb, double tola, double tolb,
                          long alpha, long beta,
                          IComplexNumber u, int ldu,
                          IComplexNumber v, int ldv,
                          IComplexNumber q, int ldq,int ncycle );

        int stgsna(int matriint, char job, char howmny,int select, int n,
                          long a, int lda,  long b,int ldb,  long vl, int ldvl,
                          long vr, int ldvr, long s,
                          long dif, int mm, int m );
        int dtgsna(int matriint, char job, char howmny,int select, int n,
                          long a, int lda,  long b,int ldb,  long vl, int ldvl,
                          long vr, int ldvr, long s,
                          long dif, int mm, int m );
        int ctgsna(int matriint, char job, char howmny,int select, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber vl, int ldvl,
                          IComplexNumber vr, int ldvr,
                          long s, long dif, int mm, int m );
        int ztgsna(int matriint, char job, char howmny,int select, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber vl, int ldvl,
                          IComplexNumber vr, int ldvr,
                          long s, long dif, int mm,int m );

        int stgsyl(int matriint, char trans, int ijob,int m, int n,  long a,int lda,  long b, int ldb,
                          long c, int ldc,  long d,int ldd,  long e, int lde,
                          long f, int ldf, long scale, long dif );
        int dtgsyl(int matriint, char trans, int ijob,int m, int n,  long a,int lda,  long b, int ldb,
                          long c, int ldc,  long d,int ldd,  long e, int lde,
                          long f, int ldf, long scale,
                          long dif );
        int ctgsyl(int matriint, char trans, int ijob,int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber c, int ldc,
                          IComplexNumber d, int ldd,
                          IComplexNumber e, int lde,
                          IComplexNumber f, int ldf,
                          long scale, long dif );
        int ztgsyl(int matriint, char trans, int ijob,int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber c, int ldc,
                          IComplexNumber d, int ldd,
                          IComplexNumber e, int lde,
                          IComplexNumber f, int ldf,
                          long scale, long dif );

        int stpcon(int matriint, char norm, char uplo, char diag,int n,  long ap, long rcond );
        int dtpcon(int matriint, char norm, char uplo, char diag,int n,  long ap, long rcond );
        int ctpcon(int matriint, char norm, char uplo, char diag,int n,  IComplexNumber ap,
                          long rcond );
        int ztpcon(int matriint, char norm, char uplo, char diag,int n,  IComplexNumber ap,
                          long rcond );

        int stprfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long ap,
                          long b, int ldb,  long x,int ldx, long ferr, long berr );
        int dtprfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long ap,
                          long b, int ldb,  long x,int ldx, long ferr, long berr );
        int ctprfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                          IComplexNumber ap,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );
        int ztprfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                          IComplexNumber ap,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );

        int stptri(int matriint, char uplo, char diag, int n,
                          long ap );
        int dtptri(int matriint, char uplo, char diag, int n,
                          long ap );
        int ctptri(int matriint, char uplo, char diag, int n,
                          IComplexNumber ap );
        int ztptri(int matriint, char uplo, char diag, int n,
                          IComplexNumber ap );

        int stptrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long ap,
                          long b, int ldb );
        int dtptrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long ap,
                          long b, int ldb );
        int ctptrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                          IComplexNumber ap,
                          IComplexNumber b, int ldb );
        int ztptrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                          IComplexNumber ap,
                          IComplexNumber b, int ldb );

        int stpttf(int matriint, char transr, char uplo,int n,  long ap, long arf );
        int dtpttf(int matriint, char transr, char uplo,int n,  long ap, long arf );
        int ctpttf(int matriint, char transr, char uplo,int n,  IComplexNumber ap,
                          IComplexNumber arf );
        int ztpttf(int matriint, char transr, char uplo,int n,  IComplexNumber ap,
                          IComplexNumber arf );

        int stpttr(int matriint, char uplo, int n,
                          long ap, long a, int lda );
        int dtpttr(int matriint, char uplo, int n,
                          long ap, long a, int lda );
        int ctpttr(int matriint, char uplo, int n,
                          IComplexNumber ap,
                          IComplexNumber a, int lda );
        int ztpttr(int matriint, char uplo, int n,
                          IComplexNumber ap,
                          IComplexNumber a, int lda );

        int strcon(int matriint, char norm, char uplo, char diag,int n,  long a, int lda,
                          long rcond );
        int dtrcon(int matriint, char norm, char uplo, char diag,int n,  long a, int lda,
                          long rcond );
        int ctrcon(int matriint, char norm, char uplo, char diag,int n,  IComplexNumber a,int lda, long rcond );
        int ztrcon(int matriint, char norm, char uplo, char diag,int n,  IComplexNumber a,int lda, long rcond );

        int strevc(int matriint, char side, char howmny,int select, int n,  long t,int ldt, long vl, int ldvl,
                          long vr, int ldvr, int mm,int m );
        int dtrevc(int matriint, char side, char howmny,int select, int n,
                          long t, int ldt, long vl,int ldvl, long vr, int ldvr,int mm, int m );
        int ctrevc(int matriint, char side, char howmny,int select, int n,
                          IComplexNumber t, int ldt,
                          IComplexNumber vl, int ldvl,
                          IComplexNumber vr, int ldvr,int mm, int m );
        int ztrevc(int matriint, char side, char howmny,int select, int n,
                          IComplexNumber t, int ldt,
                          IComplexNumber vl, int ldvl,
                          IComplexNumber vr, int ldvr,int mm, int m );

        int strexc(int matriint, char compq, int n, long t,int ldt, long q, int ldq,int ifst, int ilst );
        int dtrexc(int matriint, char compq, int n,
                          long t, int ldt, long q, int ldq,int ifst, int ilst );
        int ctrexc(int matriint, char compq, int n,
                          IComplexNumber t, int ldt,
                          IComplexNumber q, int ldq,int ifst, int ilst );
        int ztrexc(int matriint, char compq, int n,
                          IComplexNumber t, int ldt,
                          IComplexNumber q, int ldq,int ifst, int ilst );

        int strrfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long a,int lda,  long b, int ldb,
                          long x, int ldx, long ferr,
                          long berr );
        int dtrrfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long a,int lda,  long b, int ldb,
                          long x, int ldx, long ferr,
                          long berr );
        int ctrrfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );
        int ztrrfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          long ferr, long berr );

        int strsen(int matriint, char job, char compq,int select, int n, long t,int ldt, long q, int ldq, long wr,
                          long wi, int m, long s, long sep );
        int dtrsen(int matriint, char job, char compq,int select, int n,
                          long t, int ldt, long q, int ldq,
                          long wr, long wi, int m, long s,
                          long sep );
        int ctrsen(int matriint, char job, char compq,int select, int n,
                          IComplexNumber t, int ldt,
                          IComplexNumber q, int ldq,
                          IComplexNumber w, int m, long s,
                          long sep );
        int ztrsen(int matriint, char job, char compq,int select, int n,
                          IComplexNumber t, int ldt,
                          IComplexNumber q, int ldq,
                          IComplexNumber w, int m, long s,
                          long sep );

        int strsna(int matriint, char job, char howmny,int select, int n,
                          long t, int ldt,  long vl,int ldvl,  long vr, int ldvr,
                          long s, long sep, int mm, int m );
        int dtrsna(int matriint, char job, char howmny,int select, int n,
                          long t, int ldt,  long vl,int ldvl,  long vr, int ldvr,
                          long s, long sep, int mm,int m );
        int ctrsna(int matriint, char job, char howmny,int select, int n,
                          IComplexNumber t, int ldt,
                          IComplexNumber vl, int ldvl,
                          IComplexNumber vr, int ldvr,
                          long s, long sep, int mm, int m );
        int ztrsna(int matriint, char job, char howmny,int select, int n,
                          IComplexNumber t, int ldt,
                          IComplexNumber vl, int ldvl,
                          IComplexNumber vr, int ldvr,
                          long s, long sep, int mm,int m );

        int strsyl(int matriint, char trana, char tranb,int isgn, int m, int n,
                          long a, int lda,  long b,int ldb, long c, int ldc,
                          long scale );
        int dtrsyl(int matriint, char trana, char tranb,int isgn, int m, int n,
                          long a, int lda,  long b,int ldb, long c, int ldc,
                          long scale );
        int ctrsyl(int matriint, char trana, char tranb,int isgn, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber c, int ldc,
                          long scale );
        int ztrsyl(int matriint, char trana, char tranb,int isgn, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber c, int ldc,
                          long scale );

        int strtri(int matriint, char uplo, char diag, int n,
                          long a, int lda );
        int dtrtri(int matriint, char uplo, char diag, int n,
                          long a, int lda );
        int ctrtri(int matriint, char uplo, char diag, int n,
                          IComplexNumber a, int lda );
        int ztrtri(int matriint, char uplo, char diag, int n,
                          IComplexNumber a, int lda );

        int strtrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long a,int lda, long b, int ldb );
        int dtrtrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long a,int lda, long b, int ldb );
        int ctrtrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb );
        int ztrtrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb );

        int strttf(int matriint, char transr, char uplo,int n,  long a, int lda,
                          long arf );
        int dtrttf(int matriint, char transr, char uplo,int n,  long a, int lda,
                          long arf );
        int ctrttf(int matriint, char transr, char uplo,int n,  IComplexNumber a,int lda, IComplexNumber arf );
        int ztrttf(int matriint, char transr, char uplo,int n,  IComplexNumber a,int lda, IComplexNumber arf );

        int strttp(int matriint, char uplo, int n,
                          long a, int lda, long ap );
        int dtrttp(int matriint, char uplo, int n,
                          long a, int lda, long ap );
        int ctrttp(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber ap );
        int ztrttp(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber ap );

        int stzrzf(int matriint, int m, int n,
                          long a, int lda, long tau );
        int dtzrzf(int matriint, int m, int n,
                          long a, int lda, long tau );
        int ctzrzf(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );
        int ztzrzf(int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );

        int cungbr(int matriint, char vect, int m,int n, int k, IComplexNumber a,int lda,  IComplexNumber tau );
        int zungbr(int matriint, char vect, int m,int n, int k, IComplexNumber a,int lda,  IComplexNumber tau );

        int cunghr(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda,  IComplexNumber tau );
        int zunghr(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda,  IComplexNumber tau );

        int cunglq(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );
        int zunglq(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );

        int cungql(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );
        int zungql(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );

        int cungqr(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );
        int zungqr(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );

        int cungrq(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );
        int zungrq(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );

        int cungtr(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );
        int zungtr(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau );

        int cunmbr(int matriint, char vect, char side, char trans,int m, int n, int k,
                          IComplexNumber a, int lda,
                          IComplexNumber tau,
                          IComplexNumber c, int ldc );
        int zunmbr(int matriint, char vect, char side, char trans,int m, int n, int k,
                          IComplexNumber a, int lda,
                          IComplexNumber tau,
                          IComplexNumber c, int ldc );

        int cunmhr(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  IComplexNumber a,int lda,  IComplexNumber tau,
                          IComplexNumber c, int ldc );
        int zunmhr(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  IComplexNumber a,int lda,  IComplexNumber tau,
                          IComplexNumber c, int ldc );

        int cunmlq(int matriint, char side, char trans,int m, int n, int k,
                          IComplexNumber a, int lda,
                          IComplexNumber tau,
                          IComplexNumber c, int ldc );
        int zunmlq(int matriint, char side, char trans,int m, int n, int k,
                          IComplexNumber a, int lda,
                          IComplexNumber tau,
                          IComplexNumber c, int ldc );

        int cunmql(int matriint, char side, char trans,int m, int n, int k,
                          IComplexNumber a, int lda,
                          IComplexNumber tau,
                          IComplexNumber c, int ldc );
        int zunmql(int matriint, char side, char trans,int m, int n, int k,
                          IComplexNumber a, int lda,
                          IComplexNumber tau,
                          IComplexNumber c, int ldc );

        int cunmqr(int matriint, char side, char trans,int m, int n, int k,
                          IComplexNumber a, int lda,
                          IComplexNumber tau,
                          IComplexNumber c, int ldc );
        int zunmqr(int matriint, char side, char trans,int m, int n, int k,
                          IComplexNumber a, int lda,
                          IComplexNumber tau,
                          IComplexNumber c, int ldc );

        int cunmrq(int matriint, char side, char trans,int m, int n, int k,
                          IComplexNumber a, int lda,
                          IComplexNumber tau,
                          IComplexNumber c, int ldc );
        int zunmrq(int matriint, char side, char trans,int m, int n, int k,
                          IComplexNumber a, int lda,
                          IComplexNumber tau,
                          IComplexNumber c, int ldc );

        int cunmrz(int matriint, char side, char trans,int m, int n, int k,int l,  IComplexNumber a,int lda,  IComplexNumber tau,
                          IComplexNumber c, int ldc );
        int zunmrz(int matriint, char side, char trans,int m, int n, int k,int l,  IComplexNumber a,int lda,  IComplexNumber tau,
                          IComplexNumber c, int ldc );

        int cunmtr(int matriint, char side, char uplo, char trans,int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau,
                          IComplexNumber c, int ldc );
        int zunmtr(int matriint, char side, char uplo, char trans,int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau,
                          IComplexNumber c, int ldc );

        int cupgtr(int matriint, char uplo, int n,
                          IComplexNumber ap,
                          IComplexNumber tau,
                          IComplexNumber q, int ldq );
        int zupgtr(int matriint, char uplo, int n,
                          IComplexNumber ap,
                          IComplexNumber tau,
                          IComplexNumber q, int ldq );

        int cupmtr(int matriint, char side, char uplo, char trans,int m, int n,
                          IComplexNumber ap,
                          IComplexNumber tau,
                          IComplexNumber c, int ldc );
        int zupmtr(int matriint, char side, char uplo, char trans,int m, int n,
                          IComplexNumber ap,
                          IComplexNumber tau,
                          IComplexNumber c, int ldc );

        int sbdsdint(int matriint, char uplo, char compq,int n, long d, long e, long u,int ldu, long vt, int ldvt,
                            long q, int iq, long work,int iwork );
        int dbdsdint(int matriint, char uplo, char compq,int n, long d, long e, long u,int ldu, long vt, int ldvt,
                            long q, int iq, long work,int iwork );

        int sbdsvdint(int matriint, char uplo, char jobz, char range,int n, long d, long e,int vl, int vu,int il, int iu, int ns,
                             long s, long z, int ldz,
                             long work, int iwork );
        int dbdsvdint(int matriint, char uplo, char jobz, char range,int n, long d, long e,int vl, int vu,int il, int iu, int ns,
                             long s, long z, int ldz,
                             long work, int iwork );

        int sbdsqint(int matriint, char uplo, int n,int ncvt, int nru, int ncc,
                            long d, long e, long vt, int ldvt,
                            long u, int ldu, long c,int ldc, long work );
        int dbdsqint(int matriint, char uplo, int n,int ncvt, int nru, int ncc,
                            long d, long e, long vt,int ldvt, long u, int ldu,
                            long c, int ldc, long work );
        int cbdsqint(int matriint, char uplo, int n,int ncvt, int nru, int ncc,
                            long d, long e, IComplexNumber vt,int ldvt, IComplexNumber u,int ldu, IComplexNumber c,int ldc, long work );
        int zbdsqint(int matriint, char uplo, int n,int ncvt, int nru, int ncc,
                            long d, long e, IComplexNumber vt,int ldvt, IComplexNumber u,int ldu, IComplexNumber c,int ldc, long work );

        int sdisnint( char job, int m, int n,
                             long d, long sep );
        int ddisnint( char job, int m, int n,
                             long d, long sep );

        int sgbbrint(int matriint, char vect, int m,int n, int ncc, int kl,int ku, long ab, int ldab,
                            long d, long e, long q, int ldq,
                            long pt, int ldpt, long c,int ldc, long work );
        int dgbbrint(int matriint, char vect, int m,int n, int ncc, int kl,int ku, long ab, int ldab,
                            long d, long e, long q, int ldq,
                            long pt, int ldpt, long c,int ldc, long work );
        int cgbbrint(int matriint, char vect, int m,int n, int ncc, int kl,int ku, IComplexNumber ab,int ldab, long d, long e,
                            IComplexNumber q, int ldq,
                            IComplexNumber pt, int ldpt,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, long rwork );
        int zgbbrint(int matriint, char vect, int m,int n, int ncc, int kl,int ku, IComplexNumber ab,int ldab, long d, long e,
                            IComplexNumber q, int ldq,
                            IComplexNumber pt, int ldpt,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, long rwork );

        int sgbcoint(int matriint, char norm, int n,int kl, int ku,  long ab,int ldab,int ipiv,
                            float anorm, long rcond, long work,int iwork );
        int dgbcoint(int matriint, char norm, int n,int kl, int ku,  long ab,int ldab,int ipiv,
                            double anorm, long rcond, long work,int iwork );
        int cgbcoint(int matriint, char norm, int n,int kl, int ku,
                            IComplexNumber ab, int ldab,int ipiv, float anorm,
                            long rcond, IComplexNumber work,
                            long rwork );
        int zgbcoint(int matriint, char norm, int n,int kl, int ku,
                            IComplexNumber ab,int ldab,int ipiv,
                            double anorm, long rcond,
                            IComplexNumber work, long rwork );

        int sgbeqint(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c,
                            long rowcnd, long colcnd, long amax );
        int dgbeqint(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c,
                            long rowcnd, long colcnd, long amax );
        int cgbeqint(int matriint, int m, int n,int kl, int ku,
                            IComplexNumber ab, int ldab,
                            long r, long c, long rowcnd,
                            long colcnd, long amax );
        int zgbeqint(int matriint, int m, int n,int kl, int ku,
                            IComplexNumber ab,int ldab, long r, long c,
                            long rowcnd, long colcnd, long amax );

        int sgbequint(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c,
                             long rowcnd, long colcnd, long amax );
        int dgbequint(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c,
                             long rowcnd, long colcnd, long amax );
        int cgbequint(int matriint, int m, int n,int kl, int ku,
                             IComplexNumber ab,int ldab, long r, long c,
                             long rowcnd, long colcnd, long amax );
        int zgbequint(int matriint, int m, int n,int kl, int ku,
                             IComplexNumber ab,int ldab, long r, long c,
                             long rowcnd, long colcnd, long amax );

        int sgbrfint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                            long ab, int ldab,
                            long afb, int ldafb,int ipiv,  long b,int ldb, long x, int ldx,
                            long ferr, long berr, long work,int iwork );
        int dgbrfint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                            long ab, int ldab,
                            long afb, int ldafb,int ipiv,  long b,int ldb, long x, int ldx,
                            long ferr, long berr, long work,int iwork );
        int cgbrfint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                            IComplexNumber ab, int ldab,
                            IComplexNumber afb,int ldafb,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );
        int zgbrfint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                            IComplexNumber ab,int ldab,
                            IComplexNumber afb,int ldafb,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );

        int sgbrfsint(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,  long ab,int ldab,  long afb,int ldafb,int ipiv,
                             long r,  long c,  long b,int ldb, long x, int ldx,
                             long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, long work,int iwork );
        int dgbrfsint(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,  long ab,int ldab,  long afb,int ldafb,int ipiv,
                             long r,  long c,
                             long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, long work,int iwork );
        int cgbrfsint(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,
                             IComplexNumber ab,int ldab,
                             IComplexNumber afb,int ldafb,int ipiv,
                             long r,  long c,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx,
                             long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );
        int zgbrfsint(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,
                             IComplexNumber ab,int ldab,
                             IComplexNumber afb,int ldafb,int ipiv,
                             long r,  long c,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx,
                             long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );

        int sgbsint(int matriint, int n, int kl,int ku, int nrhs, long ab,int ldab, int ipiv, long b,int ldb );
        int dgbsint(int matriint, int n, int kl,int ku, int nrhs, long ab,int ldab, int ipiv, long b,int ldb );
        int cgbsint(int matriint, int n, int kl,int ku, int nrhs,
                           IComplexNumber ab, int ldab,int ipiv, IComplexNumber b,int ldb );
        int zgbsint(int matriint, int n, int kl,int ku, int nrhs,
                           IComplexNumber ab, int ldab,int ipiv, IComplexNumber b,int ldb );

        int sgbsvint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                            long afb, int ldafb, int ipiv,
                            String equed, long r, long c, long b,int ldb, long x, int ldx,
                            long rcond, long ferr, long berr,
                            long work, int iwork );
        int dgbsvint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                            long afb, int ldafb, int ipiv,
                            String equed, long r, long c, long b,int ldb, long x, int ldx,
                            long rcond, long ferr, long berr,
                            long work, int iwork );
        int cgbsvint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed,
                            long r, long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                            long berr, IComplexNumber work,
                            long rwork );
        int zgbsvint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed,
                            long r, long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                            long berr, IComplexNumber work,
                            long rwork );

        int sgbsvxint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                             long afb, int ldafb, int ipiv,
                             String equed, long r, long c, long b,int ldb, long x, int ldx,
                             long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, long work,int iwork );
        int dgbsvxint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                             long afb, int ldafb,int ipiv, String equed, long r,
                             long c, long b, int ldb,
                             long x, int ldx, long rcond,
                             long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, long work,int iwork );
        int cgbsvxint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv,
                             String equed, long r, long c,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx,
                             long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );
        int zgbsvxint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv,
                             String equed, long r, long c,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx,
                             long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );

        int sgbtrint(int matriint, int m, int n,int kl, int ku, long ab,int ldab, int ipiv );
        int dgbtrint(int matriint, int m, int n,int kl, int ku, long ab,int ldab, int ipiv );
        int cgbtrint(int matriint, int m, int n,int kl, int ku,
                            IComplexNumber ab, int ldab,int ipiv );
        int zgbtrint(int matriint, int m, int n,int kl, int ku,
                            IComplexNumber ab, int ldab,int ipiv );

        int sgbtrint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                            long ab, int ldab,int ipiv, long b,int ldb );
        int dgbtrint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                            long ab, int ldab,int ipiv, long b,int ldb );
        int cgbtrint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                            IComplexNumber ab, int ldab,int ipiv, IComplexNumber b,int ldb );
        int zgbtrint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                            IComplexNumber ab,int ldab,int ipiv,
                            IComplexNumber b, int ldb );

        int sgebaint(int matriint, char job, char side,int n, int ilo, int ihi,
                            long scale, int m, long v,int ldv );
        int dgebaint(int matriint, char job, char side,int n, int ilo, int ihi,
                            long scale, int m, long v,int ldv );
        int cgebaint(int matriint, char job, char side,int n, int ilo, int ihi,
                            long scale, int m,
                            IComplexNumber v, int ldv );
        int zgebaint(int matriint, char job, char side,int n, int ilo, int ihi,
                            long scale, int m,
                            IComplexNumber v, int ldv );

        int sgebaint(int matriint, char job, int n,
                            long a, int lda, int ilo,int ihi, long scale );
        int dgebaint(int matriint, char job, int n,
                            long a, int lda, int ilo,int ihi, long scale );
        int cgebaint(int matriint, char job, int n,
                            IComplexNumber a, int lda,int ilo, int ihi,
                            long scale );
        int zgebaint(int matriint, char job, int n,
                            IComplexNumber a, int lda,int ilo, int ihi,
                            long scale );

        int sgebrint(int matriint, int m, int n,
                            long a, int lda, long d, long e,
                            long tauq, long taup, long work,int lwork );
        int dgebrint(int matriint, int m, int n,
                            long a, int lda, long d, long e,
                            long tauq, long taup, long work,int lwork );
        int cgebrint(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            long d, long e, IComplexNumber tauq,
                            IComplexNumber taup,
                            IComplexNumber work, int lwork );
        int zgebrint(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            long d, long e,
                            IComplexNumber tauq,
                            IComplexNumber taup,
                            IComplexNumber work, int lwork );

        int sgecoint(int matriint, char norm, int n,
                            long a, int lda, float anorm,
                            long rcond, long work, int iwork );
        int dgecoint(int matriint, char norm, int n,
                            long a, int lda, double anorm,
                            long rcond, long work,int iwork );
        int cgecoint(int matriint, char norm, int n,
                            IComplexNumber a, int lda,
                            float anorm, long rcond,
                            IComplexNumber work, long rwork );
        int zgecoint(int matriint, char norm, int n,
                            IComplexNumber a, int lda,
                            double anorm, long rcond,
                            IComplexNumber work, long rwork );

        int sgeeqint(int matriint, int m, int n,
                            long a, int lda, long r,
                            long c, long rowcnd, long colcnd,
                            long amax );
        int dgeeqint(int matriint, int m, int n,
                            long a, int lda, long r,
                            long c, long rowcnd, long colcnd,
                            long amax );
        int cgeeqint(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            long r, long c, long rowcnd,
                            long colcnd, long amax );
        int zgeeqint(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            long r, long c, long rowcnd,
                            long colcnd, long amax );

        int sgeequint(int matriint, int m, int n,
                             long a, int lda, long r,
                             long c, long rowcnd, long colcnd,
                             long amax );
        int dgeequint(int matriint, int m, int n,
                             long a, int lda, long r,
                             long c, long rowcnd, long colcnd,
                             long amax );
        int cgeequint(int matriint, int m, int n,
                             IComplexNumber a, int lda,
                             long r, long c, long rowcnd,
                             long colcnd, long amax );
        int zgeequint(int matriint, int m, int n,
                             IComplexNumber a, int lda,
                             long r, long c, long rowcnd,
                             long colcnd, long amax );

        int sgeeint(int matriint, char jobvs, char sort,int select, int n, long a,int lda, int sdim, long wr,
                           long wi, long vs, int ldvs,
                           long work, int lwork,int bwork );
        int dgeeint(int matriint, char jobvs, char sort,int select, int n, long a,int lda, int sdim, long wr,
                           long wi, long vs, int ldvs,
                           long work, int lwork,int bwork );
        int cgeeint(int matriint, char jobvs, char sort,int select, int n,
                           IComplexNumber a, int lda,int sdim, IComplexNumber w,
                           IComplexNumber vs, int ldvs,
                           IComplexNumber work, int lwork,
                           long rwork, int bwork );
        int zgeeint(int matriint, char jobvs, char sort,int select, int n,
                           IComplexNumber a, int lda,int sdim, IComplexNumber w,
                           IComplexNumber vs, int ldvs,
                           IComplexNumber work, int lwork,
                           long rwork, int bwork );

        int sgeesint(int matriint, char jobvs, char sort,int select, char sense,int n, long a, int lda,int sdim, long wr, long wi,
                            long vs, int ldvs, long rconde,
                            long rcondv, long work, int lwork,int iwork, int liwork,int bwork );
        int dgeesint(int matriint, char jobvs, char sort,int select, char sense,int n, long a, int lda,int sdim, long wr, long wi,
                            long vs, int ldvs, long rconde,
                            long rcondv, long work, int lwork,int iwork, int liwork,int bwork );
        int cgeesint(int matriint, char jobvs, char sort,int select, char sense,int n, IComplexNumber a,int lda, int sdim,
                            IComplexNumber w,
                            IComplexNumber vs, int ldvs,
                            long rconde, long rcondv,
                            IComplexNumber work, int lwork,
                            long rwork, int bwork );
        int zgeesint(int matriint, char jobvs, char sort,int select, char sense,int n, IComplexNumber a,int lda, int sdim,
                            IComplexNumber w,
                            IComplexNumber vs, int ldvs,
                            long rconde, long rcondv,
                            IComplexNumber work, int lwork,
                            long rwork, int bwork );

        int sgeeint(int matriint, char jobvl, char jobvr,int n, long a, int lda,
                           long wr, long wi, long vl, int ldvl,
                           long vr, int ldvr, long work,int lwork );
        int dgeeint(int matriint, char jobvl, char jobvr,int n, long a, int lda,
                           long wr, long wi, long vl,int ldvl, long vr, int ldvr,
                           long work, int lwork );
        int cgeeint(int matriint, char jobvl, char jobvr,int n, IComplexNumber a,int lda, IComplexNumber w,
                           IComplexNumber vl, int ldvl,
                           IComplexNumber vr, int ldvr,
                           IComplexNumber work, int lwork,
                           long rwork );
        int zgeeint(int matriint, char jobvl, char jobvr,int n, IComplexNumber a,int lda, IComplexNumber w,
                           IComplexNumber vl, int ldvl,
                           IComplexNumber vr, int ldvr,
                           IComplexNumber work, int lwork,
                           long rwork );

        int sgeevint(int matriint, char balanc, char jobvl,
                            char jobvr, char sense, int n, long a,int lda, long wr, long wi, long vl,int ldvl, long vr, int ldvr,int ilo, int ihi, long scale,
                            long abnrm, long rconde, long rcondv,
                            long work, int lwork,int iwork );
        int dgeevint(int matriint, char balanc, char jobvl,
                            char jobvr, char sense, int n, long a,int lda, long wr, long wi,
                            long vl, int ldvl, long vr,int ldvr, int ilo,int ihi, long scale, long abnrm,
                            long rconde, long rcondv, long work,int lwork, int iwork );
        int cgeevint(int matriint, char balanc, char jobvl,
                            char jobvr, char sense, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber w,
                            IComplexNumber vl, int ldvl,
                            IComplexNumber vr, int ldvr,int ilo, int ihi, long scale,
                            long abnrm, long rconde, long rcondv,
                            IComplexNumber work, int lwork,
                            long rwork );
        int zgeevint(int matriint, char balanc, char jobvl,
                            char jobvr, char sense, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber w,
                            IComplexNumber vl, int ldvl,
                            IComplexNumber vr, int ldvr,int ilo, int ihi, long scale,
                            long abnrm, long rconde, long rcondv,
                            IComplexNumber work, int lwork,
                            long rwork );

        int sgehrint(int matriint, int n, int ilo,int ihi, long a, int lda,
                            long tau, long work, int lwork );
        int dgehrint(int matriint, int n, int ilo,int ihi, long a, int lda,
                            long tau, long work, int lwork );
        int cgehrint(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda, IComplexNumber tau,
                            IComplexNumber work, int lwork );
        int zgehrint(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda, IComplexNumber tau,
                            IComplexNumber work, int lwork );

        int sgejsint(int matriint, char joba, char jobu,
                            char jobv, char jobr, char jobt, char jobp,int m, int n, long a,int lda, long sva, long u,int ldu, long v, int ldv,
                            long work, int lwork,int iwork );
        int dgejsint(int matriint, char joba, char jobu,
                            char jobv, char jobr, char jobt, char jobp,int m, int n, long a,int lda, long sva, long u,int ldu, long v, int ldv,
                            long work, int lwork,int iwork );
        int cgejsint(int matriint, char joba, char jobu,
                            char jobv, char jobr, char jobt, char jobp,int m, int n, IComplexNumber a,int lda, long sva, IComplexNumber u,int ldu, IComplexNumber v, int ldv,
                            IComplexNumber cwork, int lwork,
                            long work, int lrwork,int iwork );
        int zgejsint(int matriint, char joba, char jobu,
                            char jobv, char jobr, char jobt, char jobp,int m, int n, IComplexNumber a,int lda, long sva, IComplexNumber u,int ldu, IComplexNumber v, int ldv,
                            IComplexNumber cwork, int lwork,
                            long work, int lrwork,int iwork );

        int sgelq2_work(int matriint, int m, int n,
                               long a, int lda, long tau,
                               long work );
        int dgelq2_work(int matriint, int m, int n,
                               long a, int lda, long tau,
                               long work );
        int cgelq2_work(int matriint, int m, int n,
                               IComplexNumber a, int lda,
                               IComplexNumber tau,
                               IComplexNumber work );
        int zgelq2_work(int matriint, int m, int n,
                               IComplexNumber a, int lda,
                               IComplexNumber tau,
                               IComplexNumber work );

        int sgelqint(int matriint, int m, int n,
                            long a, int lda, long tau,
                            long work, int lwork );
        int dgelqint(int matriint, int m, int n,
                            long a, int lda, long tau,
                            long work, int lwork );
        int cgelqint(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );
        int zgelqint(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );

        int sgelint(int matriint, char trans, int m,int n, int nrhs, long a,int lda, long b, int ldb,
                           long work, int lwork );
        int dgelint(int matriint, char trans, int m,int n, int nrhs, long a,int lda, long b, int ldb,
                           long work, int lwork );
        int cgelint(int matriint, char trans, int m,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,
                           IComplexNumber work, int lwork );
        int zgelint(int matriint, char trans, int m,int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,
                           IComplexNumber work, int lwork );

        int sgelsint(int matriint, int m, int n,int nrhs, long a, int lda,
                            long b, int ldb, long s, float rcond,int rank, long work, int lwork,int iwork );
        int dgelsint(int matriint, int m, int n,int nrhs, long a, int lda,
                            long b, int ldb, long s,
                            double rcond, int rank, long work,int lwork, int iwork );
        int cgelsint(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, float rcond,int rank, IComplexNumber work,int lwork, long rwork,int iwork );
        int zgelsint(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, double rcond,int rank, IComplexNumber work,int lwork, long rwork,int iwork );

        int sgelsint(int matriint, int m, int n,int nrhs, long a, int lda,
                            long b, int ldb, long s, float rcond,int rank, long work,int lwork );
        int dgelsint(int matriint, int m, int n,int nrhs, long a, int lda,
                            long b, int ldb, long s,
                            double rcond, int rank, long work,int lwork );
        int cgelsint(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, float rcond,int rank, IComplexNumber work,int lwork, long rwork );
        int zgelsint(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, double rcond,int rank, IComplexNumber work,int lwork, long rwork );

        int sgelsint(int matriint, int m, int n,int nrhs, long a, int lda,
                            long b, int ldb, int jpvt,
                            float rcond, int rank, long work,int lwork );
        int dgelsint(int matriint, int m, int n,int nrhs, long a, int lda,
                            long b, int ldb, int jpvt,
                            double rcond, int rank, long work,int lwork );
        int cgelsint(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, int jpvt, float rcond,int rank, IComplexNumber work,int lwork, long rwork );
        int zgelsint(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, int jpvt, double rcond,int rank, IComplexNumber work,int lwork, long rwork );

        int sgeqlint(int matriint, int m, int n,
                            long a, int lda, long tau,
                            long work, int lwork );
        int dgeqlint(int matriint, int m, int n,
                            long a, int lda, long tau,
                            long work, int lwork );
        int cgeqlint(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );
        int zgeqlint(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );

        int sgeqp3_work(int matriint, int m, int n,
                               long a, int lda, int jpvt,
                               long tau, long work, int lwork );
        int dgeqp3_work(int matriint, int m, int n,
                               long a, int lda, int jpvt,
                               long tau, long work, int lwork );
        int cgeqp3_work(int matriint, int m, int n,
                               IComplexNumber a, int lda,int jpvt, IComplexNumber tau,
                               IComplexNumber work, int lwork,
                               long rwork );
        int zgeqp3_work(int matriint, int m, int n,
                               IComplexNumber a, int lda,int jpvt, IComplexNumber tau,
                               IComplexNumber work, int lwork,
                               long rwork );

        int sgeqpint(int matriint, int m, int n,
                            long a, int lda, int jpvt,
                            long tau, long work );
        int dgeqpint(int matriint, int m, int n,
                            long a, int lda, int jpvt,
                            long tau, long work );
        int cgeqpint(int matriint, int m, int n,
                            IComplexNumber a, int lda,int jpvt, IComplexNumber tau,
                            IComplexNumber work, long rwork );
        int zgeqpint(int matriint, int m, int n,
                            IComplexNumber a, int lda,int jpvt, IComplexNumber tau,
                            IComplexNumber work, long rwork );

        int sgeqr2_work(int matriint, int m, int n,
                               long a, int lda, long tau,
                               long work );
        int dgeqr2_work(int matriint, int m, int n,
                               long a, int lda, long tau,
                               long work );
        int cgeqr2_work(int matriint, int m, int n,
                               IComplexNumber a, int lda,
                               IComplexNumber tau,
                               IComplexNumber work );
        int zgeqr2_work(int matriint, int m, int n,
                               IComplexNumber a, int lda,
                               IComplexNumber tau,
                               IComplexNumber work );

        int sgeqrint(int matriint, int m, int n,
                            long a, int lda, long tau,
                            long work, int lwork );
        int dgeqrint(int matriint, int m, int n,
                            long a, int lda, long tau,
                            long work, int lwork );
        int cgeqrint(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );
        int zgeqrint(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );

        int sgeqrfint(int matriint, int m, int n,
                             long a, int lda, long tau,
                             long work, int lwork );
        int dgeqrfint(int matriint, int m, int n,
                             long a, int lda, long tau,
                             long work, int lwork );
        int cgeqrfint(int matriint, int m, int n,
                             IComplexNumber a, int lda,
                             IComplexNumber tau,
                             IComplexNumber work, int lwork );
        int zgeqrfint(int matriint, int m, int n,
                             IComplexNumber a, int lda,
                             IComplexNumber tau,
                             IComplexNumber work,int lwork );

        int sgerfint(int matriint, char trans, int n,int nrhs,  long a, int lda,
                            long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                            long ferr, long berr, long work,int iwork );
        int dgerfint(int matriint, char trans, int n,int nrhs,  long a,int lda,  long af,int ldaf,int ipiv,
                            long b, int ldb, long x,int ldx, long ferr, long berr,
                            long work, int iwork );
        int cgerfint(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );
        int zgerfint(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );

        int sgerfsint(int matriint, char trans, char equed,int n, int nrhs,  long a,int lda,  long af,int ldaf,int ipiv,
                             long r,  long c,  long b,int ldb, long x, int ldx,
                             long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, long work,int iwork );
        int dgerfsint(int matriint, char trans, char equed,int n, int nrhs,  long a,int lda,  long af,int ldaf,int ipiv,
                             long r,  long c,
                             long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, long work,int iwork );
        int cgerfsint(int matriint, char trans, char equed,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af,int ldaf,int ipiv,
                             long r,  long c,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx,
                             long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );
        int zgerfsint(int matriint, char trans, char equed,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af,int ldaf,int ipiv,
                             long r,  long c,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx,
                             long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );

        int sgerqint(int matriint, int m, int n,
                            long a, int lda, long tau,
                            long work, int lwork );
        int dgerqint(int matriint, int m, int n,
                            long a, int lda, long tau,
                            long work, int lwork );
        int cgerqint(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );
        int zgerqint(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );

        int sgesdint(int matriint, char jobz, int m,int n, long a, int lda,
                            long s, long u, int ldu, long vt,int ldvt, long work, int lwork,int iwork );
        int dgesdint(int matriint, char jobz, int m,int n, long a, int lda,
                            long s, long u, int ldu,
                            long vt, int ldvt, long work,int lwork, int iwork );
        int cgesdint(int matriint, char jobz, int m,int n, IComplexNumber a,int lda, long s,
                            IComplexNumber u, int ldu,
                            IComplexNumber vt, int ldvt,
                            IComplexNumber work, int lwork,
                            long rwork, int iwork );
        int zgesdint(int matriint, char jobz, int m,int n, IComplexNumber a,int lda, long s,
                            IComplexNumber u, int ldu,
                            IComplexNumber vt, int ldvt,
                            IComplexNumber work, int lwork,
                            long rwork, int iwork );

        int sgesint(int matriint, int n, int nrhs,
                           long a, int lda, int ipiv,
                           long b, int ldb );
        int dgesint(int matriint, int n, int nrhs,
                           long a, int lda, int ipiv,
                           long b, int ldb );
        int cgesint(int matriint, int n, int nrhs,
                           IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb );
        int zgesint(int matriint, int n, int nrhs,
                           IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb );
        int dsgesint(int matriint, int n, int nrhs,
                            long a, int lda, int ipiv,
                            long b, int ldb, long x,int ldx, long work, long swork,int iter );
        int zcgesint(int matriint, int n, int nrhs,
                            IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, IComplexNumber x,int ldx, IComplexNumber work,
                            IComplexNumber swork, long rwork,int iter );

        int sgesvint(int matriint, char jobu, char jobvt,int m, int n, long a,int lda, long s, long u,int ldu, long vt, int ldvt,
                            long work, int lwork );
        int dgesvint(int matriint, char jobu, char jobvt,int m, int n, long a,int lda, long s, long u,int ldu, long vt, int ldvt,
                            long work, int lwork );
        int cgesvint(int matriint, char jobu, char jobvt,int m, int n,
                            IComplexNumber a, int lda,
                            long s, IComplexNumber u,int ldu, IComplexNumber vt,int ldvt, IComplexNumber work,int lwork, long rwork );
        int zgesvint(int matriint, char jobu, char jobvt,int m, int n,
                            IComplexNumber a, int lda,
                            long s, IComplexNumber u,int ldu, IComplexNumber vt,int ldvt, IComplexNumber work,int lwork, long rwork );

        int sgesvdint(int matriint, char jobu, char jobvt, char range,int m, int n, long a,int lda, int vl, int vu,int il, int iu, int ns,
                             long s, long u, int ldu,
                             long vt, int ldvt,
                             long work, int lwork, int iwork );
        int dgesvdint(int matriint, char jobu, char jobvt, char range,int m, int n, long a,int lda, int vl, int vu,int il, int iu, int ns,
                             long s, long u, int ldu,
                             long vt, int ldvt,
                             long work, int lwork, int iwork );
        int cgesvdint(int matriint, char jobu, char jobvt, char range,int m, int n, IComplexNumber a,int lda, int vl, int vu,int il, int iu, int ns,
                             long s, IComplexNumber u, int ldu,
                             IComplexNumber vt, int ldvt,
                             IComplexNumber work, int lwork,
                             long rwork, int iwork );
        int zgesvdint(int matriint, char jobu, char jobvt, char range,int m, int n, IComplexNumber a,int lda, int vl, int vu,int il, int iu, int ns,
                             long s, IComplexNumber u, int ldu,
                             IComplexNumber vt, int ldvt,
                             IComplexNumber work, int lwork,
                             long rwork, int iwork );

        int sgesvint(int matriint, char joba, char jobu,
                            char jobv, int m, int n, long a,int lda, long sva, int mv,
                            long v, int ldv, long work,int lwork );
        int dgesvint(int matriint, char joba, char jobu,
                            char jobv, int m, int n,
                            long a, int lda, long sva,int mv, long v, int ldv,
                            long work, int lwork );
        int cgesvint(int matriint, char joba, char jobu,
                            char jobv, int m, int n, IComplexNumber a,int lda, long sva, int mv,
                            IComplexNumber v, int ldv,
                            IComplexNumber cwork, int lwork,
                            long rwork,int lrwork );
        int zgesvint(int matriint, char joba, char jobu,
                            char jobv, int m, int n,
                            IComplexNumber a, int lda, long sva,int mv, IComplexNumber v, int ldv,
                            IComplexNumber cwork, int lwork,
                            long rwork, int lrwork );

        int sgesvint(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r,
                            long c, long b, int ldb, long x,int ldx, long rcond, long ferr,
                            long berr, long work, int iwork );
        int dgesvint(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r,
                            long c, long b, int ldb, long x,int ldx, long rcond, long ferr,
                            long berr, long work, int iwork );
        int cgesvint(int matriint, char fact, char trans,int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv, String equed, long r,
                            long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                            long berr, IComplexNumber work,
                            long rwork );
        int zgesvint(int matriint, char fact, char trans,int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv, String equed, long r,
                            long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                            long berr, IComplexNumber work,
                            long rwork );

        int sgesvxint(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r,
                             long c, long b, int ldb, long x,int ldx, long rcond, long rpvgrw,
                             long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params, long work,int iwork );
        int dgesvxint(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r,
                             long c, long b, int ldb,
                             long x, int ldx, long rcond,
                             long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, long work,int iwork );
        int cgesvxint(int matriint, char fact, char trans,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv, String equed, long r,
                             long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long rpvgrw,
                             long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork );
        int zgesvxint(int matriint, char fact, char trans,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv, String equed, long r,
                             long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long rpvgrw,
                             long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork );

        int sgetf2_work(int matriint, int m, int n,
                               long a, int lda, int ipiv );
        int dgetf2_work(int matriint, int m, int n,
                               long a, int lda, int ipiv );
        int cgetf2_work(int matriint, int m, int n,
                               IComplexNumber a, int lda,int ipiv );
        int zgetf2_work(int matriint, int m, int n,
                               IComplexNumber a, int lda,int ipiv );

        int sgetrint(int matriint, int m, int n,
                            long a, int lda, int ipiv );
        int dgetrint(int matriint, int m, int n,
                            long a, int lda, int ipiv );
        int cgetrint(int matriint, int m, int n,
                            IComplexNumber a, int lda,int ipiv );
        int zgetrint(int matriint, int m, int n,
                            IComplexNumber a, int lda,int ipiv );

        int sgetrf2_work(int matriint, int m, int n,
                                long a, int lda, int ipiv );
        int dgetrf2_work(int matriint, int m, int n,
                                long a, int lda, int ipiv );
        int cgetrf2_work(int matriint, int m, int n,
                                IComplexNumber a, int lda,int ipiv );
        int zgetrf2_work(int matriint, int m, int n,
                                IComplexNumber a, int lda,int ipiv );

        int sgetrint(int matriint, int n, long a,int lda,int ipiv,
                            long work, int lwork );
        int dgetrint(int matriint, int n, long a,int lda,int ipiv,
                            long work, int lwork );
        int cgetrint(int matriint, int n,
                            IComplexNumber a, int lda,int ipiv,
                            IComplexNumber work, int lwork );
        int zgetrint(int matriint, int n,
                            IComplexNumber a, int lda,int ipiv,
                            IComplexNumber work, int lwork );

        int sgetrint(int matriint, char trans, int n,int nrhs,  long a, int lda,int ipiv, long b,int ldb );
        int dgetrint(int matriint, char trans, int n,int nrhs,  long a,int lda,int ipiv,
                            long b, int ldb );
        int cgetrint(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                            IComplexNumber b, int ldb );
        int zgetrint(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                            IComplexNumber b, int ldb );

        int sggbaint(int matriint, char job, char side,int n, int ilo, int ihi,
                            long lscale,  long rscale,int m, long v, int ldv );
        int dggbaint(int matriint, char job, char side,int n, int ilo, int ihi,
                            long lscale,  long rscale,int m, long v, int ldv );
        int cggbaint(int matriint, char job, char side,int n, int ilo, int ihi,
                            long lscale,  long rscale,int m, IComplexNumber v,int ldv );
        int zggbaint(int matriint, char job, char side,int n, int ilo, int ihi,
                            long lscale,  long rscale,int m, IComplexNumber v,int ldv );

        int sggbaint(int matriint, char job, int n,
                            long a, int lda, long b,int ldb, int ilo,int ihi, long lscale, long rscale,
                            long work );
        int dggbaint(int matriint, char job, int n,
                            long a, int lda, long b,int ldb, int ilo,int ihi, long lscale, long rscale,
                            long work );
        int cggbaint(int matriint, char job, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,int ilo, int ihi, long lscale,
                            long rscale, long work );
        int zggbaint(int matriint, char job, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,int ilo, int ihi,
                            long lscale, long rscale, long work );

        int sggeint(int matriint, char jobvsl, char jobvsr,
                           char sort, int selctg, int n,
                           long a, int lda, long b,int ldb, int sdim, long alphar,
                           long alphai, long beta, long vsl,int ldvsl, long vsr, int ldvsr,
                           long work, int lwork,int bwork );
        int dggeint(int matriint, char jobvsl, char jobvsr,
                           char sort, int selctg, int n,
                           long a, int lda, long b,int ldb, int sdim, long alphar,
                           long alphai, long beta, long vsl,int ldvsl, long vsr, int ldvsr,
                           long work, int lwork,int bwork );
        int cggeint(int matriint, char jobvsl, char jobvsr,
                           char sort, int selctg, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                           IComplexNumber beta,
                           IComplexNumber vsl, int ldvsl,
                           IComplexNumber vsr, int ldvsr,
                           IComplexNumber work, int lwork,
                           long rwork, int bwork );
        int zggeint(int matriint, char jobvsl, char jobvsr,
                           char sort, int selctg, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                           IComplexNumber beta,
                           IComplexNumber vsl, int ldvsl,
                           IComplexNumber vsr, int ldvsr,
                           IComplexNumber work, int lwork,
                           long rwork, int bwork );

        int sgges3_work(int matriint, char jobvsl, char jobvsr,
                               char sort, int selctg,int n,
                               long a, int lda,
                               long b, int ldb, int sdim,
                               long alphar, long alphai, long beta,
                               long vsl, int ldvsl,
                               long vsr, int ldvsr,
                               long work, int lwork,int bwork );
        int dgges3_work(int matriint, char jobvsl, char jobvsr,
                               char sort, int selctg,int n,
                               long a, int lda,
                               long b, int ldb, int sdim,
                               long alphar, long alphai, long beta,
                               long vsl, int ldvsl,
                               long vsr, int ldvsr,
                               long work, int lwork,int bwork );
        int cgges3_work(int matriint, char jobvsl, char jobvsr,
                               char sort, int selctg,int n,
                               IComplexNumber a, int lda,
                               IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                               IComplexNumber beta,
                               IComplexNumber vsl, int ldvsl,
                               IComplexNumber vsr, int ldvsr,
                               IComplexNumber work, int lwork,
                               long rwork, int bwork );
        int zgges3_work(int matriint, char jobvsl, char jobvsr,
                               char sort, int selctg,int n,
                               IComplexNumber a, int lda,
                               IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                               IComplexNumber beta,
                               IComplexNumber vsl, int ldvsl,
                               IComplexNumber vsr, int ldvsr,
                               IComplexNumber work, int lwork,
                               long rwork, int bwork );

        int sggesint(int matriint, char jobvsl, char jobvsr,
                            char sort, int selctg, char sense,int n, long a, int lda,
                            long b, int ldb, int sdim,
                            long alphar, long alphai, long beta,
                            long vsl, int ldvsl, long vsr,int ldvsr, long rconde, long rcondv,
                            long work, int lwork,int iwork, int liwork,int bwork );
        int dggesint(int matriint, char jobvsl, char jobvsr,
                            char sort, int selctg, char sense,int n, long a, int lda,
                            long b, int ldb, int sdim,
                            long alphar, long alphai, long beta,
                            long vsl, int ldvsl, long vsr,int ldvsr, long rconde,
                            long rcondv, long work, int lwork,int iwork, int liwork,int bwork );
        int cggesint(int matriint, char jobvsl, char jobvsr,
                            char sort, int selctg, char sense,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, int sdim,
                            IComplexNumber alpha,
                            IComplexNumber beta,
                            IComplexNumber vsl, int ldvsl,
                            IComplexNumber vsr, int ldvsr,
                            long rconde, long rcondv,
                            IComplexNumber work, int lwork,
                            long rwork, int iwork,int liwork, int bwork );
        int zggesint(int matriint, char jobvsl, char jobvsr,
                            char sort, int selctg, char sense,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, int sdim,
                            IComplexNumber alpha,
                            IComplexNumber beta,
                            IComplexNumber vsl, int ldvsl,
                            IComplexNumber vsr, int ldvsr,
                            long rconde, long rcondv,
                            IComplexNumber work, int lwork,
                            long rwork, int iwork,int liwork, int bwork );

        int sggeint(int matriint, char jobvl, char jobvr,int n, long a, int lda, long b,int ldb, long alphar, long alphai,
                           long beta, long vl, int ldvl,
                           long vr, int ldvr, long work,int lwork );
        int dggeint(int matriint, char jobvl, char jobvr,int n, long a, int lda,
                           long b, int ldb, long alphar,
                           long alphai, long beta, long vl,int ldvl, long vr, int ldvr,
                           long work, int lwork );
        int cggeint(int matriint, char jobvl, char jobvr,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber alpha,
                           IComplexNumber beta,
                           IComplexNumber vl, int ldvl,
                           IComplexNumber vr, int ldvr,
                           IComplexNumber work, int lwork,
                           long rwork );
        int zggeint(int matriint, char jobvl, char jobvr,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber alpha,
                           IComplexNumber beta,
                           IComplexNumber vl, int ldvl,
                           IComplexNumber vr, int ldvr,
                           IComplexNumber work, int lwork,
                           long rwork );

        int sggev3_work(int matriint, char jobvl, char jobvr,int n,
                               long a, int lda,
                               long b, int ldb,
                               long alphar, long alphai, long beta,
                               long vl, int ldvl,
                               long vr, int ldvr,
                               long work, int lwork );
        int dggev3_work(int matriint, char jobvl, char jobvr,int n,
                               long a, int lda,
                               long b, int ldb,
                               long alphar, long alphai, long beta,
                               long vl, int ldvl,
                               long vr, int ldvr,
                               long work, int lwork );
        int cggev3_work(int matriint, char jobvl, char jobvr,int n,
                               IComplexNumber a, int lda,
                               IComplexNumber b, int ldb,
                               IComplexNumber alpha,
                               IComplexNumber beta,
                               IComplexNumber vl, int ldvl,
                               IComplexNumber vr, int ldvr,
                               IComplexNumber work, int lwork,
                               long rwork );
        int zggev3_work(int matriint, char jobvl, char jobvr,int n,
                               IComplexNumber a, int lda,
                               IComplexNumber b, int ldb,
                               IComplexNumber alpha,
                               IComplexNumber beta,
                               IComplexNumber vl, int ldvl,
                               IComplexNumber vr, int ldvr,
                               IComplexNumber work, int lwork,
                               long rwork );

        int sggevint(int matriint, char balanc, char jobvl,
                            char jobvr, char sense, int n, long a,int lda, long b, int ldb,
                            long alphar, long alphai, long beta,
                            long vl, int ldvl, long vr,int ldvr, int ilo,int ihi, long lscale, long rscale,
                            long abnrm, long bbnrm, long rconde,
                            long rcondv, long work, int lwork,int iwork, int bwork );
        int dggevint(int matriint, char balanc, char jobvl,
                            char jobvr, char sense, int n, long a,int lda, long b, int ldb,
                            long alphar, long alphai, long beta,
                            long vl, int ldvl, long vr,int ldvr, int ilo,int ihi, long lscale, long rscale,
                            long abnrm, long bbnrm, long rconde,
                            long rcondv, long work, int lwork,int iwork, int bwork );
        int cggevint(int matriint, char balanc, char jobvl,
                            char jobvr, char sense, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber alpha,
                            IComplexNumber beta,
                            IComplexNumber vl, int ldvl,
                            IComplexNumber vr, int ldvr,int ilo, int ihi, long lscale,
                            long rscale, long abnrm, long bbnrm,
                            long rconde, long rcondv,
                            IComplexNumber work, int lwork,
                            long rwork, int iwork,int bwork );
        int zggevint(int matriint, char balanc, char jobvl,
                            char jobvr, char sense, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber alpha,
                            IComplexNumber beta,
                            IComplexNumber vl, int ldvl,
                            IComplexNumber vr, int ldvr,int ilo, int ihi,
                            long lscale, long rscale, long abnrm,
                            long bbnrm, long rconde, long rcondv,
                            IComplexNumber work, int lwork,
                            long rwork, int iwork,int bwork );

        int sggglint(int matriint, int n, int m,int p, long a, int lda,
                            long b, int ldb, long d, long x,
                            long y, long work, int lwork );
        int dggglint(int matriint, int n, int m,int p, long a, int lda,
                            long b, int ldb, long d, long x,
                            long y, long work, int lwork );
        int cggglint(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber d,
                            IComplexNumber x,
                            IComplexNumber y,
                            IComplexNumber work, int lwork );
        int zggglint(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber d,
                            IComplexNumber x,
                            IComplexNumber y,
                            IComplexNumber work, int lwork );

        int sgghrint(int matriint, char compq, char compz,int n, int ilo, int ihi,
                            long a, int lda, long b,int ldb, long q, int ldq,
                            long z, int ldz );
        int dgghrint(int matriint, char compq, char compz,int n, int ilo, int ihi,
                            long a, int lda, long b,int ldb, long q, int ldq,
                            long z, int ldz );
        int cgghrint(int matriint, char compq, char compz,int n, int ilo, int ihi,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber q, int ldq,
                            IComplexNumber z, int ldz );
        int zgghrint(int matriint, char compq, char compz,int n, int ilo, int ihi,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber q, int ldq,
                            IComplexNumber z, int ldz );

        int sgghd3_work(int matriint, char compq, char compz,int n, int ilo, int ihi,
                               long a, int lda,
                               long b, int ldb,
                               long q, int ldq,
                               long z, int ldz,
                               long work, int lwork );
        int dgghd3_work(int matriint, char compq, char compz,int n, int ilo, int ihi,
                               long a, int lda,
                               long b, int ldb,
                               long q, int ldq,
                               long z, int ldz,
                               long work, int lwork );
        int cgghd3_work(int matriint, char compq, char compz,int n, int ilo, int ihi,
                               IComplexNumber a, int lda,
                               IComplexNumber b, int ldb,
                               IComplexNumber q, int ldq,
                               IComplexNumber z, int ldz,
                               IComplexNumber work, int lwork );
        int zgghd3_work(int matriint, char compq, char compz,int n, int ilo, int ihi,
                               IComplexNumber a, int lda,
                               IComplexNumber b, int ldb,
                               IComplexNumber q, int ldq,
                               IComplexNumber z, int ldz,
                               IComplexNumber work,int lwork );

        int sgglsint(int matriint, int m, int n,int p, long a, int lda,
                            long b, int ldb, long c, long d,
                            long x, long work, int lwork );
        int dgglsint(int matriint, int m, int n,int p, long a, int lda,
                            long b, int ldb, long c, long d,
                            long x, long work, int lwork );
        int cgglsint(int matriint, int m, int n,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber c,
                            IComplexNumber d,
                            IComplexNumber x,
                            IComplexNumber work, int lwork );
        int zgglsint(int matriint, int m, int n,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber c,
                            IComplexNumber d,
                            IComplexNumber x,
                            IComplexNumber work, int lwork );

        int sggqrint(int matriint, int n, int m,int p, long a, int lda,
                            long taua, long b, int ldb,
                            long taub, long work, int lwork );
        int dggqrint(int matriint, int n, int m,int p, long a, int lda,
                            long taua, long b, int ldb,
                            long taub, long work, int lwork );
        int cggqrint(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber taua,
                            IComplexNumber b, int ldb,
                            IComplexNumber taub,
                            IComplexNumber work, int lwork );
        int zggqrint(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber taua,
                            IComplexNumber b, int ldb,
                            IComplexNumber taub,
                            IComplexNumber work, int lwork );

        int sggrqint(int matriint, int m, int p,int n, long a, int lda,
                            long taua, long b, int ldb,
                            long taub, long work, int lwork );
        int dggrqint(int matriint, int m, int p,int n, long a, int lda,
                            long taua, long b, int ldb,
                            long taub, long work, int lwork );
        int cggrqint(int matriint, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber taua,
                            IComplexNumber b, int ldb,
                            IComplexNumber taub,
                            IComplexNumber work, int lwork );
        int zggrqint(int matriint, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber taua,
                            IComplexNumber b, int ldb,
                            IComplexNumber taub,
                            IComplexNumber work, int lwork );

        int sggsvint(int matriint, char jobu, char jobv,
                            char jobq, int m, int n,int p, int k, int l,
                            long a, int lda, long b,int ldb, long alpha, long beta,
                            long u, int ldu, long v,int ldv, long q, int ldq,
                            long work, int iwork );
        int dggsvint(int matriint, char jobu, char jobv,
                            char jobq, int m, int n,int p, int k, int l,
                            long a, int lda, long b,int ldb, long alpha, long beta,
                            long u, int ldu, long v,int ldv, long q, int ldq,
                            long work, int iwork );
        int cggsvint(int matriint, char jobu, char jobv,
                            char jobq, int m, int n,int p, int k, int l,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            long alpha, long beta,
                            IComplexNumber u, int ldu,
                            IComplexNumber v, int ldv,
                            IComplexNumber q, int ldq,
                            IComplexNumber work, long rwork,int iwork );
        int zggsvint(int matriint, char jobu, char jobv,
                            char jobq, int m, int n,int p, int k, int l,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            long alpha, long beta,
                            IComplexNumber u, int ldu,
                            IComplexNumber v, int ldv,
                            IComplexNumber q, int ldq,
                            IComplexNumber work, long rwork,int iwork );

        int sggsvd3_work(int matriint, char jobu, char jobv,
                                char jobq, int m, int n,int p, int k, int l,
                                long a, int lda, long b,int ldb, long alpha, long beta,
                                long u, int ldu, long v,int ldv, long q, int ldq,
                                long work, int lwork,int iwork );
        int dggsvd3_work(int matriint, char jobu, char jobv,
                                char jobq, int m, int n,int p, int k, int l,
                                long a, int lda, long b,int ldb, long alpha, long beta,
                                long u, int ldu, long v,int ldv, long q, int ldq,
                                long work, int lwork,int iwork );
        int cggsvd3_work(int matriint, char jobu, char jobv,
                                char jobq, int m, int n,int p, int k, int l,
                                IComplexNumber a, int lda,
                                IComplexNumber b, int ldb,
                                long alpha, long beta,
                                IComplexNumber u, int ldu,
                                IComplexNumber v, int ldv,
                                IComplexNumber q, int ldq,
                                IComplexNumber work, int lwork,
                                long rwork, int iwork );
        int zggsvd3_work(int matriint, char jobu, char jobv,
                                char jobq, int m, int n,int p, int k, int l,
                                IComplexNumber a, int lda,
                                IComplexNumber b, int ldb,
                                long alpha, long beta,
                                IComplexNumber u, int ldu,
                                IComplexNumber v, int ldv,
                                IComplexNumber q, int ldq,
                                IComplexNumber work, int lwork,
                                long rwork, int iwork );

        int sggsvint(int matriint, char jobu, char jobv,
                            char jobq, int m, int p,int n, long a, int lda,
                            long b, int ldb, float tola,
                            float tolb, int k, int l,
                            long u, int ldu, long v,int ldv, long q, int ldq,int iwork, long tau, long work );
        int dggsvint(int matriint, char jobu, char jobv,
                            char jobq, int m, int p,int n, long a, int lda,
                            long b, int ldb, double tola,
                            double tolb, int k, int l,
                            long u, int ldu, long v,int ldv, long q, int ldq,int iwork, long tau, long work );
        int cggsvint(int matriint, char jobu, char jobv,
                            char jobq, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, float tola, float tolb,int k, int l,
                            IComplexNumber u, int ldu,
                            IComplexNumber v, int ldv,
                            IComplexNumber q, int ldq,int iwork, long rwork,
                            IComplexNumber tau,
                            IComplexNumber work );
        int zggsvint(int matriint, char jobu, char jobv,
                            char jobq, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, double tola, double tolb,int k, int l,
                            IComplexNumber u, int ldu,
                            IComplexNumber v, int ldv,
                            IComplexNumber q, int ldq,int iwork, long rwork,
                            IComplexNumber tau,
                            IComplexNumber work );

        int sggsvp3_work(int matriint, char jobu, char jobv,
                                char jobq, int m, int p,int n, long a, int lda,
                                long b, int ldb, float tola,
                                float tolb, int k, int l,
                                long u, int ldu, long v,int ldv, long q, int ldq,int iwork, long tau,
                                long work, int lwork );
        int dggsvp3_work(int matriint, char jobu, char jobv,
                                char jobq, int m, int p,int n, long a, int lda,
                                long b, int ldb, double tola,
                                double tolb, int k, int l,
                                long u, int ldu, long v,int ldv, long q, int ldq,int iwork, long tau, long work,int lwork );
        int cggsvp3_work(int matriint, char jobu, char jobv,
                                char jobq, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, float tola, float tolb,int k, int l,
                                IComplexNumber u, int ldu,
                                IComplexNumber v, int ldv,
                                IComplexNumber q, int ldq,int iwork, long rwork,
                                IComplexNumber tau,
                                IComplexNumber work, int lwork );
        int zggsvp3_work(int matriint, char jobu, char jobv,
                                char jobq, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, double tola, double tolb,int k, int l,
                                IComplexNumber u, int ldu,
                                IComplexNumber v, int ldv,
                                IComplexNumber q, int ldq,int iwork, long rwork,
                                IComplexNumber tau,
                                IComplexNumber work, int lwork );

        int sgtcoint( char norm, int n,  long dl,
                             long d,  long du,
                             long du2,int ipiv,
                             float anorm, long rcond, long work,int iwork );
        int dgtcoint( char norm, int n,  long dl,
                             long d,  long du,
                             long du2,int ipiv,
                             double anorm, long rcond, long work,int iwork );
        int cgtcoint( char norm, int n,
                             IComplexNumber dl,
                             IComplexNumber d,
                             IComplexNumber du,
                             IComplexNumber du2,int ipiv, float anorm,
                             long rcond, IComplexNumber work );
        int zgtcoint( char norm, int n,
                             IComplexNumber dl,
                             IComplexNumber d,
                             IComplexNumber du,
                             IComplexNumber du2,int ipiv, double anorm,
                             long rcond, IComplexNumber work );

        int sgtrfint(int matriint, char trans, int n,int nrhs,  long dl,
                            long d,  long du,
                            long dlf,  long df,
                            long duf,  long du2,int ipiv,  long b,int ldb, long x, int ldx,
                            long ferr, long berr, long work,int iwork );
        int dgtrfint(int matriint, char trans, int n,int nrhs,  long dl,
                            long d,  long du,
                            long dlf,  long df,
                            long duf,  long du2,int ipiv,  long b,int ldb, long x, int ldx,
                            long ferr, long berr, long work,int iwork );
        int cgtrfint(int matriint, char trans, int n,int nrhs,  IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du,
                            IComplexNumber dlf,
                            IComplexNumber df,
                            IComplexNumber duf,
                            IComplexNumber du2,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );
        int zgtrfint(int matriint, char trans, int n,int nrhs,
                            IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du,
                            IComplexNumber dlf,
                            IComplexNumber df,
                            IComplexNumber duf,
                            IComplexNumber du2,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );

        int sgtsint(int matriint, int n, int nrhs,
                           long dl, long d, long du, long b,int ldb );
        int dgtsint(int matriint, int n, int nrhs,
                           long dl, long d, long du, long b,int ldb );
        int cgtsint(int matriint, int n, int nrhs,
                           IComplexNumber dl,
                           IComplexNumber d,
                           IComplexNumber du,
                           IComplexNumber b, int ldb );
        int zgtsint(int matriint, int n, int nrhs,
                           IComplexNumber dl,
                           IComplexNumber d,
                           IComplexNumber du,
                           IComplexNumber b, int ldb );

        int sgtsvint(int matriint, char fact, char trans,int n, int nrhs,  long dl,
                            long d,  long du, long dlf,
                            long df, long duf, long du2,int ipiv,  long b,int ldb, long x, int ldx,
                            long rcond, long ferr, long berr,
                            long work, int iwork );
        int dgtsvint(int matriint, char fact, char trans,int n, int nrhs,  long dl,
                            long d,  long du, long dlf,
                            long df, long duf, long du2,int ipiv,  long b,int ldb, long x, int ldx,
                            long rcond, long ferr, long berr,
                            long work, int iwork );
        int cgtsvint(int matriint, char fact, char trans,int n, int nrhs,
                            IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du,
                            IComplexNumber dlf,
                            IComplexNumber df,
                            IComplexNumber duf,
                            IComplexNumber du2, int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork );
        int zgtsvint(int matriint, char fact, char trans,int n, int nrhs,
                            IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du,
                            IComplexNumber dlf,
                            IComplexNumber df,
                            IComplexNumber duf,
                            IComplexNumber du2, int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork );

        int sgttrint(int n, long dl, long d, long du,
                            long du2, int ipiv );
        int dgttrint(int n, long dl, long d, long du,
                            long du2, int ipiv );
        int cgttrint(int n, IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du,
                            IComplexNumber du2, int ipiv );
        int zgttrint(int n, IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du,
                            IComplexNumber du2, int ipiv );

        int sgttrint(int matriint, char trans, int n,int nrhs,  long dl,
                            long d,  long du,
                            long du2,int ipiv,
                            long b, int ldb );
        int dgttrint(int matriint, char trans, int n,int nrhs,  long dl,
                            long d,  long du,
                            long du2,int ipiv,
                            long b, int ldb );
        int cgttrint(int matriint, char trans, int n,int nrhs,  IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du,
                            IComplexNumber du2,int ipiv, IComplexNumber b,int ldb );
        int zgttrint(int matriint, char trans, int n,int nrhs,
                            IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du,
                            IComplexNumber du2,int ipiv,
                            IComplexNumber b, int ldb );

        int chbeint(int matriint, char jobz, char uplo,int n, int kd,
                           IComplexNumber ab, int ldab,
                           long w, IComplexNumber z,int ldz, IComplexNumber work,
                           long rwork );
        int zhbeint(int matriint, char jobz, char uplo,int n, int kd,
                           IComplexNumber ab, int ldab,
                           long w, IComplexNumber z,int ldz, IComplexNumber work,
                           long rwork );

        int chbevint(int matriint, char jobz, char uplo,int n, int kd,
                            IComplexNumber ab, int ldab,
                            long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );
        int zhbevint(int matriint, char jobz, char uplo,int n, int kd,
                            IComplexNumber ab, int ldab,
                            long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );

        int chbevint(int matriint, char jobz, char range,
                            char uplo, int n, int kd,
                            IComplexNumber ab, int ldab,
                            IComplexNumber q, int ldq,
                            float vl, float vu, int il,int iu, float abstol, int m,
                            long w, IComplexNumber z,int ldz, IComplexNumber work,
                            long rwork, int iwork,int ifail );
        int zhbevint(int matriint, char jobz, char range,
                            char uplo, int n, int kd,
                            IComplexNumber ab, int ldab,
                            IComplexNumber q, int ldq,
                            double vl, double vu, int il,int iu, double abstol, int m,
                            long w, IComplexNumber z,int ldz, IComplexNumber work,
                            long rwork, int iwork,int ifail );

        int chbgsint(int matriint, char vect, char uplo,int n, int ka, int kb,
                            IComplexNumber ab, int ldab,
                            IComplexNumber bb, int ldbb,
                            IComplexNumber x, int ldx,
                            IComplexNumber work, long rwork );
        int zhbgsint(int matriint, char vect, char uplo,int n, int ka, int kb,
                            IComplexNumber ab, int ldab,
                            IComplexNumber bb,int ldbb, IComplexNumber x,int ldx, IComplexNumber work,
                            long rwork );

        int chbgint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                           IComplexNumber ab, int ldab,
                           IComplexNumber bb, int ldbb,
                           long w, IComplexNumber z,int ldz, IComplexNumber work,
                           long rwork );
        int zhbgint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                           IComplexNumber ab, int ldab,
                           IComplexNumber bb, int ldbb,
                           long w, IComplexNumber z,int ldz, IComplexNumber work,
                           long rwork );

        int chbgvint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                            IComplexNumber ab, int ldab,
                            IComplexNumber bb, int ldbb,
                            long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );
        int zhbgvint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                            IComplexNumber ab, int ldab,
                            IComplexNumber bb, int ldbb,
                            long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );

        int chbgvint(int matriint, char jobz, char range,
                            char uplo, int n, int ka,int kb, IComplexNumber ab,int ldab, IComplexNumber bb,int ldbb, IComplexNumber q,int ldq, float vl, float vu,int il, int iu, float abstol,int m, long w,
                            IComplexNumber z, int ldz,
                            IComplexNumber work, long rwork,int iwork, int ifail );
        int zhbgvint(int matriint, char jobz, char range,
                            char uplo, int n, int ka,int kb, IComplexNumber ab,int ldab, IComplexNumber bb,int ldbb, IComplexNumber q,int ldq, double vl, double vu,int il, int iu, double abstol,int m, long w,
                            IComplexNumber z, int ldz,
                            IComplexNumber work, long rwork,int iwork, int ifail );

        int chbtrint(int matriint, char vect, char uplo,int n, int kd,
                            IComplexNumber ab, int ldab,
                            long d, long e, IComplexNumber q,int ldq, IComplexNumber work );
        int zhbtrint(int matriint, char vect, char uplo,int n, int kd,
                            IComplexNumber ab, int ldab,
                            long d, long e, IComplexNumber q,int ldq, IComplexNumber work );

        int checoint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv, float anorm,
                            long rcond, IComplexNumber work );
        int zhecoint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv, double anorm,
                            long rcond, IComplexNumber work );

        int cheequint(int matriint, char uplo, int n,
                             IComplexNumber a, int lda,
                             long s, long scond, long amax,
                             IComplexNumber work );
        int zheequint(int matriint, char uplo, int n,
                             IComplexNumber a, int lda,
                             long s, long scond, long amax,
                             IComplexNumber work );

        int cheeint(int matriint, char jobz, char uplo,int n, IComplexNumber a,int lda, long w,
                           IComplexNumber work, int lwork,
                           long rwork );
        int zheeint(int matriint, char jobz, char uplo,int n, IComplexNumber a,int lda, long w,
                           IComplexNumber work, int lwork,
                           long rwork );

        int cheevint(int matriint, char jobz, char uplo,int n, IComplexNumber a,int lda, long w,
                            IComplexNumber work, int lwork,
                            long rwork, int lrwork,int iwork, int liwork );
        int zheevint(int matriint, char jobz, char uplo,int n, IComplexNumber a,int lda, long w,
                            IComplexNumber work, int lwork,
                            long rwork, int lrwork,int iwork, int liwork );

        int cheevint(int matriint, char jobz, char range,
                            char uplo, int n,
                            IComplexNumber a, int lda,
                            float vl, float vu, int il,int iu, float abstol, int m,
                            long w, IComplexNumber z,int ldz, int isuppz,
                            IComplexNumber work, int lwork,
                            long rwork, int lrwork,int iwork, int liwork );
        int zheevint(int matriint, char jobz, char range,
                            char uplo, int n,
                            IComplexNumber a, int lda,
                            double vl, double vu, int il,int iu, double abstol, int m,
                            long w, IComplexNumber z,int ldz, int isuppz,
                            IComplexNumber work, int lwork,
                            long rwork, int lrwork,int iwork, int liwork );

        int cheevint(int matriint, char jobz, char range,
                            char uplo, int n,
                            IComplexNumber a, int lda,
                            float vl, float vu, int il,int iu, float abstol, int m,
                            long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int iwork, int ifail );
        int zheevint(int matriint, char jobz, char range,
                            char uplo, int n,
                            IComplexNumber a, int lda,
                            double vl, double vu, int il,int iu, double abstol, int m,
                            long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int iwork, int ifail );

        int chegsint(int matriint, int itype, char uplo,int n, IComplexNumber a,int lda,  IComplexNumber b,int ldb );
        int zhegsint(int matriint, int itype, char uplo,int n, IComplexNumber a,int lda,  IComplexNumber b,int ldb );

        int chegint(int matriint, int itype, char jobz,
                           char uplo, int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, long w,
                           IComplexNumber work, int lwork,
                           long rwork );
        int zhegint(int matriint, int itype, char jobz,
                           char uplo, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,
                           long w, IComplexNumber work,int lwork, long rwork );

        int chegvint(int matriint, int itype, char jobz,
                            char uplo, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            long w, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );
        int zhegvint(int matriint, int itype, char jobz,
                            char uplo, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            long w, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );

        int chegvint(int matriint, int itype, char jobz,
                            char range, char uplo, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            float vl, float vu, int il,int iu, float abstol, int m,
                            long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int iwork, int ifail );
        int zhegvint(int matriint, int itype, char jobz,
                            char range, char uplo, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            double vl, double vu, int il,int iu, double abstol, int m,
                            long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int iwork, int ifail );

        int cherfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );
        int zherfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );

        int cherfsint(int matriint, char uplo, char equed,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af,int ldaf,int ipiv,
                             long s,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );
        int zherfsint(int matriint, char uplo, char equed,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af,int ldaf,int ipiv,
                             long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx,
                             long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );

        int chesint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                           IComplexNumber b, int ldb,
                           IComplexNumber work, int lwork );
        int zhesint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                           IComplexNumber b, int ldb,
                           IComplexNumber work, int lwork );

        int chesvint(int matriint, char fact, char uplo,int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                            long berr, IComplexNumber work,int lwork, long rwork );
        int zhesvint(int matriint, char fact, char uplo,int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, int lwork,
                            long rwork );

        int chesvxint(int matriint, char fact, char uplo,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx,
                             long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );
        int zhesvxint(int matriint, char fact, char uplo,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx,
                             long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );

        int chetrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,
                            long d, long e, IComplexNumber tau,
                            IComplexNumber work, int lwork );
        int zhetrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,
                            long d, long e,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );

        int chetrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );
        int zhetrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );

        int chetrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv,
                            IComplexNumber work );
        int zhetrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv,
                            IComplexNumber work );


        int chfrint(int matriint, char transr, char uplo,
                           char trans, int n, int k,
                           float alpha,  IComplexNumber a,int lda, float beta,
                           IComplexNumber c );
        int zhfrint(int matriint, char transr, char uplo,
                           char trans, int n, int k,
                           double alpha,  IComplexNumber a,int lda, double beta,
                           IComplexNumber c );

        int shgeqint(int matriint, char job, char compq,
                            char compz, int n, int ilo,int ihi, long h, int ldh,
                            long t, int ldt, long alphar,
                            long alphai, long beta, long q,int ldq, long z, int ldz,
                            long work, int lwork );
        int dhgeqint(int matriint, char job, char compq,
                            char compz, int n, int ilo,int ihi, long h, int ldh,
                            long t, int ldt, long alphar,
                            long alphai, long beta, long q,int ldq, long z, int ldz,
                            long work, int lwork );
        int chgeqint(int matriint, char job, char compq,
                            char compz, int n, int ilo,int ihi, IComplexNumber h,int ldh, IComplexNumber t,int ldt, IComplexNumber alpha,
                            IComplexNumber beta,
                            IComplexNumber q, int ldq,
                            IComplexNumber z, int ldz,
                            IComplexNumber work, int lwork,
                            long rwork );
        int zhgeqint(int matriint, char job, char compq,
                            char compz, int n, int ilo,int ihi, IComplexNumber h,int ldh, IComplexNumber t,int ldt, IComplexNumber alpha,
                            IComplexNumber beta,
                            IComplexNumber q, int ldq,
                            IComplexNumber z, int ldz,
                            IComplexNumber work, int lwork,
                            long rwork );

        int chpcoint(int matriint, char uplo, int n,
                            IComplexNumber ap,int ipiv, float anorm,
                            long rcond, IComplexNumber work );
        int zhpcoint(int matriint, char uplo, int n,
                            IComplexNumber ap,int ipiv, double anorm,
                            long rcond, IComplexNumber work );

        int chpeint(int matriint, char jobz, char uplo,int n, IComplexNumber ap, long w,
                           IComplexNumber z, int ldz,
                           IComplexNumber work, long rwork );
        int zhpeint(int matriint, char jobz, char uplo,int n, IComplexNumber ap,
                           long w, IComplexNumber z,int ldz, IComplexNumber work,
                           long rwork );

        int chpevint(int matriint, char jobz, char uplo,int n, IComplexNumber ap,
                            long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );
        int zhpevint(int matriint, char jobz, char uplo,int n, IComplexNumber ap,
                            long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );

        int chpevint(int matriint, char jobz, char range,
                            char uplo, int n,
                            IComplexNumber ap, float vl, float vu,int il, int iu, float abstol,int m, long w,
                            IComplexNumber z, int ldz,
                            IComplexNumber work, long rwork,int iwork, int ifail );
        int zhpevint(int matriint, char jobz, char range,
                            char uplo, int n,
                            IComplexNumber ap, double vl, double vu,int il, int iu, double abstol,int m, long w,
                            IComplexNumber z, int ldz,
                            IComplexNumber work, long rwork,int iwork, int ifail );

        int chpgsint(int matriint, int itype, char uplo,int n, IComplexNumber ap,
                            IComplexNumber bp );
        int zhpgsint(int matriint, int itype, char uplo,int n, IComplexNumber ap,
                            IComplexNumber bp );

        int chpgint(int matriint, int itype, char jobz,
                           char uplo, int n,
                           IComplexNumber ap,
                           IComplexNumber bp, long w,
                           IComplexNumber z, int ldz,
                           IComplexNumber work, long rwork );
        int zhpgint(int matriint, int itype, char jobz,
                           char uplo, int n,
                           IComplexNumber ap,
                           IComplexNumber bp, long w,
                           IComplexNumber z, int ldz,
                           IComplexNumber work, long rwork );

        int chpgvint(int matriint, int itype, char jobz,
                            char uplo, int n,
                            IComplexNumber ap,
                            IComplexNumber bp, long w,
                            IComplexNumber z, int ldz,
                            IComplexNumber work, int lwork,
                            long rwork, int lrwork,int iwork, int liwork );
        int zhpgvint(int matriint, int itype, char jobz,
                            char uplo, int n,
                            IComplexNumber ap,
                            IComplexNumber bp, long w,
                            IComplexNumber z, int ldz,
                            IComplexNumber work, int lwork,
                            long rwork, int lrwork,int iwork, int liwork );

        int chpgvint(int matriint, int itype, char jobz,
                            char range, char uplo, int n,
                            IComplexNumber ap,
                            IComplexNumber bp, float vl, float vu,int il, int iu, float abstol,int m, long w,
                            IComplexNumber z, int ldz,
                            IComplexNumber work, long rwork,int iwork, int ifail );
        int zhpgvint(int matriint, int itype, char jobz,
                            char range, char uplo, int n,
                            IComplexNumber ap,
                            IComplexNumber bp, double vl, double vu,int il, int iu, double abstol,int m, long w,
                            IComplexNumber z, int ldz,
                            IComplexNumber work, long rwork,int iwork, int ifail );

        int chprfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                            IComplexNumber afp,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );
        int zhprfint(int matriint, char uplo, int n,int nrhs,
                            IComplexNumber ap,
                            IComplexNumber afp,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );

        int chpsint(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
        int zhpsint(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );

        int chpsvint(int matriint, char fact, char uplo,int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber afp, int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork );
        int zhpsvint(int matriint, char fact, char uplo,int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber afp, int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork );

        int chptrint(int matriint, char uplo, int n,
                            IComplexNumber ap, long d, long e,
                            IComplexNumber tau );
        int zhptrint(int matriint, char uplo, int n,
                            IComplexNumber ap, long d, long e,
                            IComplexNumber tau );

        int chptrint(int matriint, char uplo, int n,
                            IComplexNumber ap, int ipiv );
        int zhptrint(int matriint, char uplo, int n,
                            IComplexNumber ap, int ipiv );

        int chptrint(int matriint, char uplo, int n,
                            IComplexNumber ap,int ipiv,
                            IComplexNumber work );
        int zhptrint(int matriint, char uplo, int n,
                            IComplexNumber ap,int ipiv,
                            IComplexNumber work );

        int chptrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
        int zhptrint(int matriint, char uplo, int n,int nrhs,
                            IComplexNumber ap,int ipiv,
                            IComplexNumber b, int ldb );

        int shseiint(int matriint, char job, char eigsrc,
                            char initv, int select,int n,  long h, int ldh,
                            long wr,  long wi, long vl,int ldvl, long vr, int ldvr,int mm, int m, long work,int ifaill, int ifailr );
        int dhseiint(int matriint, char job, char eigsrc,
                            char initv, int select,int n,  long h, int ldh,
                            long wr,  long wi, long vl,int ldvl, long vr, int ldvr,int mm, int m, long work,int ifaill, int ifailr );
        int chseiint(int matriint, char job, char eigsrc,
                            char initv,int select,int n,  IComplexNumber h,int ldh, IComplexNumber w,
                            IComplexNumber vl, int ldvl,
                            IComplexNumber vr, int ldvr,int mm, int m,
                            IComplexNumber work, long rwork,int ifaill, int ifailr );
        int zhseiint(int matriint, char job, char eigsrc,
                            char initv,int select,int n,  IComplexNumber h,int ldh, IComplexNumber w,
                            IComplexNumber vl, int ldvl,
                            IComplexNumber vr, int ldvr,int mm, int m,
                            IComplexNumber work, long rwork,int ifaill, int ifailr );

        int shseqint(int matriint, char job, char compz,int n, int ilo, int ihi,
                            long h, int ldh, long wr, long wi,
                            long z, int ldz, long work,int lwork );
        int dhseqint(int matriint, char job, char compz,int n, int ilo, int ihi,
                            long h, int ldh, long wr,
                            long wi, long z, int ldz,
                            long work, int lwork );
        int chseqint(int matriint, char job, char compz,int n, int ilo, int ihi,
                            IComplexNumber h, int ldh,
                            IComplexNumber w,
                            IComplexNumber z, int ldz,
                            IComplexNumber work, int lwork );
        int zhseqint(int matriint, char job, char compz,int n, int ilo, int ihi,
                            IComplexNumber h, int ldh,
                            IComplexNumber w,
                            IComplexNumber z, int ldz,
                            IComplexNumber work, int lwork );

        int clacgint(int n, IComplexNumber x,int incx );
        int zlacgint(int n, IComplexNumber x,int incx );

        int slacn2_work(int n, long v, long x,int isgn, long est, int kase,int isave );
        int dlacn2_work(int n, long v, long x,int isgn, long est, int kase,int isave );
        int clacn2_work(int n, IComplexNumber v,
                               IComplexNumber x,
                               long est, int kase,int isave );
        int zlacn2_work(int n, IComplexNumber v,
                               IComplexNumber x,
                               long est, int kase,int isave );

        int slacpint(int matriint, char uplo, int m,int n,  long a, int lda,
                            long b, int ldb );
        int dlacpint(int matriint, char uplo, int m,int n,  long a, int lda,
                            long b, int ldb );
        int clacpint(int matriint, char uplo, int m,int n,  IComplexNumber a,int lda, IComplexNumber b,int ldb );
        int zlacpint(int matriint, char uplo, int m,int n,  IComplexNumber a,int lda, IComplexNumber b,int ldb );

        int clacp2_work(int matriint, char uplo, int m,int n,  long a, int lda,
                               IComplexNumber b, int ldb );
        int zlacp2_work(int matriint, char uplo, int m,int n,  long a, int lda,
                               IComplexNumber b, int ldb );

        int zlag2int(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber sa, int ldsa );

        int slag2int(int matriint, int m, int n,
                            long sa, int ldsa, long a,int lda );

        int dlag2int(int matriint, int m, int n,
                            long a, int lda, long sa,int ldsa );

        int clag2int(int matriint, int m, int n,
                            IComplexNumber sa, int ldsa,
                            IComplexNumber a, int lda );

        int slaggint(int matriint, int m, int n,int kl, int ku,  long d,
                            long a, int lda, int iseed,
                            long work );
        int dlaggint(int matriint, int m, int n,int kl, int ku,  long d,
                            long a, int lda, int iseed,
                            long work );
        int claggint(int matriint, int m, int n,int kl, int ku,  long d,
                            IComplexNumber a, int lda,int iseed, IComplexNumber work );
        int zlaggint(int matriint, int m, int n,int kl, int ku,  long d,
                            IComplexNumber a, int lda,int iseed,
                            IComplexNumber work );

        int claghint(int matriint, int n, int k,
                            long d, IComplexNumber a,int lda, int iseed,
                            IComplexNumber work );
        int zlaghint(int matriint, int n, int k,
                            long d, IComplexNumber a,int lda, int iseed,
                            IComplexNumber work );

        int slagsint(int matriint, int n, int k,
                            long d, long a, int lda,int iseed, long work );
        int dlagsint(int matriint, int n, int k,
                            long d, long a, int lda,int iseed, long work );
        int clagsint(int matriint, int n, int k,
                            long d, IComplexNumber a,int lda, int iseed,
                            IComplexNumber work );
        int zlagsint(int matriint, int n, int k,
                            long d, IComplexNumber a,int lda, int iseed,
                            IComplexNumber work );

        int slapmint(int matriint, int forwrd,int m, int n, long x,int ldx, int k );
        int dlapmint(int matriint, int forwrd,int m, int n, long x,int ldx, int k );
        int clapmint(int matriint, int forwrd,int m, int n,
                            IComplexNumber x, int ldx,int k );
        int zlapmint(int matriint, int forwrd,int m, int n,
                            IComplexNumber x, int ldx,int k );


        int slartgint( float f, float g, long cs, long sn,
                              long r );
        int dlartgint( double f, double g, long cs, long sn,
                              long r );

        int slartgint( float x, float y, float sigma, long cs,
                              long sn );
        int dlartgint( double x, double y, double sigma, long cs,
                              long sn );

        float slapy2_work( float x, float y );
        double dlapy2_work( double x, double y );

        float slapy3_work( float x, float y, float z );
         double dlapy3_work( double x, double y, double z );

          float slamcint( char cmach );
         double dlamcint( char cmach );

        float slangint(int matriint, char norm, int m,int n,  long a, int lda,
                              long work );
        double dlangint(int matriint, char norm, int m,int n,  long a, int lda,
                               long work );
        float clangint(int matriint, char norm, int m,int n,  IComplexNumber a,int lda, long work );
        double zlangint(int matriint, char norm, int m,int n,  IComplexNumber a,int lda, long work );

        float clanhint(int matriint, char norm, char uplo,int n,  IComplexNumber a,int lda, long work );
        double zlanhint(int matriint, char norm, char uplo,int n,  IComplexNumber a,int lda, long work );

        float slansint(int matriint, char norm, char uplo,int n,  long a, int lda,
                              long work );
         double dlansint(int matriint, char norm, char uplo,int n,  long a, int lda,
                                long work );
         float clansint(int matriint, char norm, char uplo,int n,  IComplexNumber a,int lda, long work );
         double zlansint(int matriint, char norm, char uplo,int n,  IComplexNumber a,int lda, long work );

        float slantint(int matriint, char norm, char uplo,
                              char diag, int m, int n,  long a,int lda, long work );
         double dlantint(int matriint, char norm, char uplo,
                                char diag, int m, int n,
                                long a, int lda, long work );
        float clantint(int matriint, char norm, char uplo,
                              char diag, int m, int n,
                              IComplexNumber a, int lda,
                              long work );
        double zlantint(int matriint, char norm, char uplo,
                               char diag, int m, int n,
                               IComplexNumber a, int lda,
                               long work );

        int slarfint(int matriint, char side, char trans,
                            char direct, char storev, int m,int n, int k,  long v,int ldv,  long t, int ldt,
                            long c, int ldc, long work,int ldwork );
        int dlarfint(int matriint, char side, char trans,
                            char direct, char storev, int m,int n, int k,  long v,int ldv,  long t, int ldt,
                            long c, int ldc, long work,int ldwork );
        int clarfint(int matriint, char side, char trans,
                            char direct, char storev, int m,int n, int k,
                            IComplexNumber v, int ldv,
                            IComplexNumber t, int ldt,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int ldwork );
        int zlarfint(int matriint, char side, char trans,
                            char direct, char storev, int m,int n, int k,
                            IComplexNumber v, int ldv,
                            IComplexNumber t, int ldt,
                            IComplexNumber c, int ldc,
                            IComplexNumber work,int ldwork );

        int slarfint(int n, long alpha, long x,int incx, long tau );
        int dlarfint(int n, long alpha, long x,int incx, long tau );
        int clarfint(int n, IComplexNumber alpha,
                            IComplexNumber x, int incx,
                            IComplexNumber tau );
        int zlarfint(int n, IComplexNumber alpha,
                            IComplexNumber x, int incx,
                            IComplexNumber tau );

        int slarfint(int matriint, char direct, char storev,int n, int k,  long v,int ldv,  long tau, long t,int ldt );
        int dlarfint(int matriint, char direct, char storev,int n, int k,  long v,int ldv,  long tau, long t,int ldt );
        int clarfint(int matriint, char direct, char storev,int n, int k,
                            IComplexNumber v, int ldv,
                            IComplexNumber tau,
                            IComplexNumber t, int ldt );
        int zlarfint(int matriint, char direct, char storev,int n, int k,
                            IComplexNumber v, int ldv,
                            IComplexNumber tau,
                            IComplexNumber t, int ldt );

        int slarfint(int matriint, char side, int m,int n,  long v, float tau,
                            long c, int ldc, long work );
        int dlarfint(int matriint, char side, int m,int n,  long v, double tau,
                            long c, int ldc, long work );
        int clarfint(int matriint, char side, int m,int n,  IComplexNumber v,
                            IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work );
        int zlarfint(int matriint, char side, int m,int n,  IComplexNumber v,
                            IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work );

        int slarnint(int idist, int iseed,int n, long x );
        int dlarnint(int idist, int iseed,int n, long x );
        int clarnint(int idist, int iseed,int n, IComplexNumber x );
        int zlarnint(int idist, int iseed,int n, IComplexNumber x );


        int slascint(int matriint, char type, int kl,int ku, float cfrom, float cto,int m, int n, long a,int lda );
        int dlascint(int matriint, char type, int kl,int ku, double cfrom, double cto,int m, int n, long a,int lda );
        int clascint(int matriint, char type, int kl,int ku, float cfrom, float cto,int m, int n, IComplexNumber a,int lda );
        int zlascint(int matriint, char type, int kl,int ku, double cfrom, double cto,int m, int n, IComplexNumber a,int lda );

        int slaseint(int matriint, char uplo, int m,int n, float alpha, float beta, long a,int lda );
        int dlaseint(int matriint, char uplo, int m,int n, double alpha, double beta,
                            long a, int lda );
        int claseint(int matriint, char uplo, int m,int n, IComplexNumber alpha,
                            IComplexNumber beta,
                            IComplexNumber a, int lda );
        int zlaseint(int matriint, char uplo, int m,int n, IComplexNumber alpha,
                            IComplexNumber beta,
                            IComplexNumber a, int lda );

        int slasrint( char id, int n, long d );
        int dlasrint( char id, int n, long d );

        int slaswint(int matriint, int n, long a,int lda, int k1, int k2,int ipiv, int incx );
        int dlaswint(int matriint, int n, long a,int lda, int k1, int k2,int ipiv, int incx );
        int claswint(int matriint, int n,
                            IComplexNumber a, int lda,int k1, int k2,int ipiv, int incx );
        int zlaswint(int matriint, int n,
                            IComplexNumber a, int lda,int k1, int k2,int ipiv, int incx );

        int slatmint(int matriint, int m, int n,
                            char dist, int iseed, char sym,
                            long d, int mode, float cond,
                            float dmax, int kl, int ku,
                            char pack, long a, int lda,
                            long work );
        int dlatmint(int matriint, int m, int n,
                            char dist, int iseed, char sym,
                            long d, int mode, double cond,
                            double dmax, int kl, int ku,
                            char pack, long a, int lda,
                            long work );
        int clatmint(int matriint, int m, int n,
                            char dist, int iseed, char sym,
                            long d, int mode, float cond,
                            float dmax, int kl, int ku,
                            char pack, IComplexNumber a,int lda, IComplexNumber work );
        int zlatmint(int matriint, int m, int n,
                            char dist, int iseed, char sym,
                            long d, int mode, double cond,
                            double dmax, int kl, int ku,
                            char pack, IComplexNumber a,int lda, IComplexNumber work );

        int slauuint(int matriint, char uplo, int n,
                            long a, int lda );
        int dlauuint(int matriint, char uplo, int n,
                            long a, int lda );
        int clauuint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda );
        int zlauuint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda );

        int sopgtint(int matriint, char uplo, int n,
                            long ap,  long tau, long q,int ldq, long work );
        int dopgtint(int matriint, char uplo, int n,
                            long ap,  long tau, long q,int ldq, long work );

        int sopmtint(int matriint, char side, char uplo,
                            char trans, int m, int n,
                            long ap,  long tau, long c,int ldc, long work );
        int dopmtint(int matriint, char side, char uplo,
                            char trans, int m, int n,
                            long ap,  long tau, long c,int ldc, long work );

        int sorgbint(int matriint, char vect, int m,int n, int k, long a,int lda,  long tau, long work,int lwork );
        int dorgbint(int matriint, char vect, int m,int n, int k, long a,int lda,  long tau, long work,int lwork );

        int sorghint(int matriint, int n, int ilo,int ihi, long a, int lda,
                            long tau, long work,int lwork );
        int dorghint(int matriint, int n, int ilo,int ihi, long a, int lda,
                            long tau, long work,int lwork );

        int sorglint(int matriint, int m, int n,int k, long a, int lda,
                            long tau, long work,int lwork );
        int dorglint(int matriint, int m, int n,int k, long a, int lda,
                            long tau, long work,int lwork );

        int sorgqint(int matriint, int m, int n,int k, long a, int lda,
                            long tau, long work,int lwork );
        int dorgqint(int matriint, int m, int n,int k, long a, int lda,
                            long tau, long work,int lwork );



        int sorgrint(int matriint, int m, int n,int k, long a, int lda,
                            long tau, long work,int lwork );
        int dorgrint(int matriint, int m, int n,int k, long a, int lda,
                            long tau, long work,int lwork );

        int sorgtint(int matriint, char uplo, int n,
                            long a, int lda,  long tau,
                            long work, int lwork );
        int dorgtint(int matriint, char uplo, int n,
                            long a, int lda,  long tau,
                            long work, int lwork );

        int sormbint(int matriint, char vect, char side,
                            char trans, int m, int n,int k,  long a, int lda,
                            long tau, long c, int ldc,
                            long work, int lwork );
        int dormbint(int matriint, char vect, char side,
                            char trans, int m, int n,int k,  long a, int lda,
                            long tau, long c, int ldc,
                            long work, int lwork );

        int sormhint(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  long a, int lda,
                            long tau, long c, int ldc,
                            long work, int lwork );
        int dormhint(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  long a, int lda,
                            long tau, long c, int ldc,
                            long work, int lwork );

        int sormlint(int matriint, char side, char trans,int m, int n, int k,
                            long a, int lda,
                            long tau, long c, int ldc,
                            long work, int lwork );
        int dormlint(int matriint, char side, char trans,int m, int n, int k,
                            long a, int lda,
                            long tau, long c, int ldc,
                            long work, int lwork );

        int sormqint(int matriint, char side, char trans,int m, int n, int k,
                            long a, int lda,
                            long tau, long c, int ldc,
                            long work, int lwork );
        int dormqint(int matriint, char side, char trans,int m, int n, int k,
                            long a, int lda,
                            long tau, long c, int ldc,
                            long work, int lwork );



        int sormrint(int matriint, char side, char trans,int m, int n, int k,
                            long a, int lda,
                            long tau, long c, int ldc,
                            long work, int lwork );
        int dormrint(int matriint, char side, char trans,int m, int n, int k,
                            long a, int lda,
                            long tau, long c, int ldc,
                            long work, int lwork );

        int sormrint(int matriint, char side, char trans,int m, int n, int k,int l,  long a, int lda,
                            long tau, long c, int ldc,
                            long work, int lwork );
        int dormrint(int matriint, char side, char trans,int m, int n, int k,int l,  long a, int lda,
                            long tau, long c, int ldc,
                            long work, int lwork );

        int sormtint(int matriint, char side, char uplo,
                            char trans, int m, int n,
                            long a, int lda,
                            long tau, long c, int ldc,
                            long work, int lwork );
        int dormtint(int matriint, char side, char uplo,
                            char trans, int m, int n,
                            long a, int lda,
                            long tau, long c, int ldc,
                            long work, int lwork );

        int spbcoint(int matriint, char uplo, int n,int kd,  long ab, int ldab,
                            float anorm, long rcond, long work,int iwork );
        int dpbcoint(int matriint, char uplo, int n,int kd,  long ab,int ldab, double anorm, long rcond,
                            long work, int iwork );
        int cpbcoint(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, float anorm, long rcond,
                            IComplexNumber work, long rwork );
        int zpbcoint(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, double anorm, long rcond,
                            IComplexNumber work, long rwork );

        int spbeqint(int matriint, char uplo, int n,int kd,  long ab, int ldab,
                            long s, long scond, long amax );
        int dpbeqint(int matriint, char uplo, int n,int kd,  long ab,int ldab, long s, long scond,
                            long amax );
        int cpbeqint(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, long s, long scond,
                            long amax );
        int zpbeqint(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, long s, long scond,
                            long amax );

        int spbrfint(int matriint, char uplo, int n,int kd, int nrhs,  long ab,int ldab,  long afb,int ldafb,  long b,int ldb, long x, int ldx,
                            long ferr, long berr, long work,int iwork );
        int dpbrfint(int matriint, char uplo, int n,int kd, int nrhs,
                            long ab, int ldab,
                            long afb, int ldafb,
                            long b, int ldb, long x,int ldx, long ferr, long berr,
                            long work, int iwork );
        int cpbrfint(int matriint, char uplo, int n,int kd, int nrhs,
                            IComplexNumber ab, int ldab,
                            IComplexNumber afb,int ldafb,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long ferr, long berr,
                            IComplexNumber work, long rwork );
        int zpbrfint(int matriint, char uplo, int n,int kd, int nrhs,
                            IComplexNumber ab,int ldab,
                            IComplexNumber afb,int ldafb,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );

        int spbstint(int matriint, char uplo, int n,int kb, long bb, int ldbb );
        int dpbstint(int matriint, char uplo, int n,int kb, long bb, int ldbb );
        int cpbstint(int matriint, char uplo, int n,int kb, IComplexNumber bb,int ldbb );
        int zpbstint(int matriint, char uplo, int n,int kb, IComplexNumber bb,int ldbb );

        int spbsint(int matriint, char uplo, int n,int kd, int nrhs, long ab,int ldab, long b, int ldb );
        int dpbsint(int matriint, char uplo, int n,int kd, int nrhs, long ab,int ldab, long b, int ldb );
        int cpbsint(int matriint, char uplo, int n,int kd, int nrhs,
                           IComplexNumber ab, int ldab,
                           IComplexNumber b, int ldb );
        int zpbsint(int matriint, char uplo, int n,int kd, int nrhs,
                           IComplexNumber ab, int ldab,
                           IComplexNumber b, int ldb );

        int spbsvint(int matriint, char fact, char uplo,int n, int kd, int nrhs,
                            long ab, int ldab, long afb,int ldafb, String equed, long s,
                            long b, int ldb, long x,int ldx, long rcond, long ferr,
                            long berr, long work, int iwork );
        int dpbsvint(int matriint, char fact, char uplo,int n, int kd, int nrhs,
                            long ab, int ldab, long afb,int ldafb, String equed, long s,
                            long b, int ldb, long x,int ldx, long rcond, long ferr,
                            long berr, long work, int iwork );
        int cpbsvint(int matriint, char fact, char uplo,int n, int kd, int nrhs,
                            IComplexNumber ab, int ldab,
                            IComplexNumber afb, int ldafb,
                            String equed, long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                            long berr, IComplexNumber work,
                            long rwork );
        int zpbsvint(int matriint, char fact, char uplo,int n, int kd, int nrhs,
                            IComplexNumber ab, int ldab,
                            IComplexNumber afb, int ldafb,
                            String equed, long s,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork );

        int spbtrint(int matriint, char uplo, int n,int kd, long ab, int ldab );
        int dpbtrint(int matriint, char uplo, int n,int kd, long ab, int ldab );
        int cpbtrint(int matriint, char uplo, int n,int kd, IComplexNumber ab,int ldab );
        int zpbtrint(int matriint, char uplo, int n,int kd, IComplexNumber ab,int ldab );

        int spbtrint(int matriint, char uplo, int n,int kd, int nrhs,  long ab,int ldab, long b, int ldb );
        int dpbtrint(int matriint, char uplo, int n,int kd, int nrhs,
                            long ab, int ldab, long b,int ldb );
        int cpbtrint(int matriint, char uplo, int n,int kd, int nrhs,
                            IComplexNumber ab, int ldab,
                            IComplexNumber b, int ldb );
        int zpbtrint(int matriint, char uplo, int n,int kd, int nrhs,
                            IComplexNumber ab,int ldab, IComplexNumber b,int ldb );

        int spftrint(int matriint, char transr, char uplo,int n, long a );
        int dpftrint(int matriint, char transr, char uplo,int n, long a );
        int cpftrint(int matriint, char transr, char uplo,int n, IComplexNumber a );
        int zpftrint(int matriint, char transr, char uplo,int n, IComplexNumber a );



        int spftrint(int matriint, char transr, char uplo,int n, int nrhs,  long a,
                            long b, int ldb );
        int dpftrint(int matriint, char transr, char uplo,int n, int nrhs,  long a,
                            long b, int ldb );
        int cpftrint(int matriint, char transr, char uplo,int n, int nrhs,
                            IComplexNumber a,
                            IComplexNumber b, int ldb );
        int zpftrint(int matriint, char transr, char uplo,int n, int nrhs,
                            IComplexNumber a,
                            IComplexNumber b, int ldb );

        int spocoint(int matriint, char uplo, int n,
                            long a, int lda, float anorm,
                            long rcond, long work, int iwork );
        int dpocoint(int matriint, char uplo, int n,
                            long a, int lda, double anorm,
                            long rcond, long work,int iwork );
        int cpocoint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,
                            float anorm, long rcond,
                            IComplexNumber work, long rwork );
        int zpocoint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,
                            double anorm, long rcond,
                            IComplexNumber work, long rwork );

        int spoeqint(int matriint, int n,  long a,int lda, long s, long scond,
                            long amax );
        int dpoeqint(int matriint, int n,  long a,int lda, long s, long scond,
                            long amax );
        int cpoeqint(int matriint, int n,
                            IComplexNumber a, int lda,
                            long s, long scond, long amax );
        int zpoeqint(int matriint, int n,
                            IComplexNumber a, int lda,
                            long s, long scond, long amax );

        int spoequint(int matriint, int n,  long a,int lda, long s, long scond,
                             long amax );
        int dpoequint(int matriint, int n,
                             long a, int lda, long s,
                             long scond, long amax );
        int cpoequint(int matriint, int n,
                             IComplexNumber a, int lda,
                             long s, long scond, long amax );
        int zpoequint(int matriint, int n,
                             IComplexNumber a, int lda,
                             long s, long scond, long amax );

        int sporfint(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                            long af, int ldaf,
                            long b, int ldb, long x,int ldx, long ferr, long berr,
                            long work, int iwork );
        int dporfint(int matriint, char uplo, int n,int nrhs,  long a,int lda,  long af,int ldaf,  long b,int ldb, long x, int ldx,
                            long ferr, long berr, long work,int iwork );
        int cporfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long ferr, long berr,
                            IComplexNumber work, long rwork );
        int zporfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long ferr, long berr,
                            IComplexNumber work, long rwork );

        int sporfsint(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af,int ldaf,  long s,
                             long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, long work,int iwork );
        int dporfsint(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af,int ldaf,  long s,
                             long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, long work,int iwork );
        int cporfsint(int matriint, char uplo, char equed,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af,int ldaf,  long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx,
                             long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );
        int zporfsint(int matriint, char uplo, char equed,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af,int ldaf,  long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx,
                             long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );

        int sposint(int matriint, char uplo, int n,int nrhs, long a, int lda,
                           long b, int ldb );
        int dposint(int matriint, char uplo, int n,int nrhs, long a, int lda,
                           long b, int ldb );
        int cposint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb );
        int zposint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb );
        int dsposint(int matriint, char uplo, int n,int nrhs, long a, int lda,
                            long b, int ldb, long x,int ldx, long work, long swork,int iter );
        int zcposint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber x,int ldx, IComplexNumber work,
                            IComplexNumber swork, long rwork,int iter );

        int sposvint(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,
                            String equed, long s, long b, int ldb,
                            long x, int ldx, long rcond,
                            long ferr, long berr, long work,int iwork );
        int dposvint(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,
                            String equed, long s, long b,int ldb, long x, int ldx,
                            long rcond, long ferr, long berr,
                            long work, int iwork );
        int cposvint(int matriint, char fact, char uplo,int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,
                            String equed, long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                            long berr, IComplexNumber work,
                            long rwork );
        int zposvint(int matriint, char fact, char uplo,int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,
                            String equed, long s,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork );

        int sposvxint(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,
                             String equed, long s, long b,int ldb, long x, int ldx,
                             long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, long work,int iwork );
        int dposvxint(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,
                             String equed, long s, long b,int ldb, long x, int ldx,
                             long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, long work,int iwork );
        int cposvxint(int matriint, char fact, char uplo,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,
                             String equed, long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long rpvgrw,
                             long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork );
        int zposvxint(int matriint, char fact, char uplo,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,
                             String equed, long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx,
                             long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );

        int spotrf2_work(int matriint, char uplo, int n,
                                long a, int lda );
        int dpotrf2_work(int matriint, char uplo, int n,
                                long a, int lda );
        int cpotrf2_work(int matriint, char uplo, int n,
                                IComplexNumber a, int lda );
        int zpotrf2_work(int matriint, char uplo, int n,
                                IComplexNumber a, int lda );

        int spotrint(int matriint, char uplo, int n,
                            long a, int lda );
        int dpotrint(int matriint, char uplo, int n,
                            long a, int lda );
        int cpotrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda );
        int zpotrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda );



        int spotrint(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                            long b, int ldb );
        int dpotrint(int matriint, char uplo, int n,int nrhs,  long a,int lda, long b, int ldb );
        int cpotrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber b,int ldb );
        int zpotrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber b,int ldb );

        int sppcoint(int matriint, char uplo, int n,
                            long ap, float anorm, long rcond,
                            long work, int iwork );
        int dppcoint(int matriint, char uplo, int n,
                            long ap, double anorm, long rcond,
                            long work, int iwork );
        int cppcoint(int matriint, char uplo, int n,
                            IComplexNumber ap, float anorm,
                            long rcond, IComplexNumber work,
                            long rwork );
        int zppcoint(int matriint, char uplo, int n,
                            IComplexNumber ap, double anorm,
                            long rcond, IComplexNumber work,
                            long rwork );

        int sppeqint(int matriint, char uplo, int n,
                            long ap, long s, long scond,
                            long amax );
        int dppeqint(int matriint, char uplo, int n,
                            long ap, long s, long scond,
                            long amax );
        int cppeqint(int matriint, char uplo, int n,
                            IComplexNumber ap, long s,
                            long scond, long amax );
        int zppeqint(int matriint, char uplo, int n,
                            IComplexNumber ap, long s,
                            long scond, long amax );

        int spprfint(int matriint, char uplo, int n,int nrhs,  long ap,
                            long afp,  long b,int ldb, long x, int ldx,
                            long ferr, long berr, long work,int iwork );
        int dpprfint(int matriint, char uplo, int n,int nrhs,  long ap,
                            long afp,  long b,int ldb, long x, int ldx,
                            long ferr, long berr, long work,int iwork );
        int cpprfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                            IComplexNumber afp,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );
        int zpprfint(int matriint, char uplo, int n,int nrhs,
                            IComplexNumber ap,
                            IComplexNumber afp,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );

        int sppsint(int matriint, char uplo, int n,int nrhs, long ap, long b,int ldb );
        int dppsint(int matriint, char uplo, int n,int nrhs, long ap, long b,int ldb );
        int cppsint(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,
                           IComplexNumber b, int ldb );
        int zppsint(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,
                           IComplexNumber b, int ldb );

        int sppsvint(int matriint, char fact, char uplo,int n, int nrhs, long ap,
                            long afp, String equed, long s, long b,int ldb, long x, int ldx,
                            long rcond, long ferr, long berr,
                            long work, int iwork );
        int dppsvint(int matriint, char fact, char uplo,int n, int nrhs, long ap,
                            long afp, String equed, long s, long b,int ldb, long x, int ldx,
                            long rcond, long ferr, long berr,
                            long work, int iwork );
        int cppsvint(int matriint, char fact, char uplo,int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber afp, String equed,
                            long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                            long berr, IComplexNumber work,
                            long rwork );
        int zppsvint(int matriint, char fact, char uplo,int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber afp, String equed,
                            long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                            long berr, IComplexNumber work,
                            long rwork );

        int spptrint(int matriint, char uplo, int n,
                            long ap );
        int dpptrint(int matriint, char uplo, int n,
                            long ap );
        int cpptrint(int matriint, char uplo, int n,
                            IComplexNumber ap );
        int zpptrint(int matriint, char uplo, int n,
                            IComplexNumber ap );



        int spptrint(int matriint, char uplo, int n,int nrhs,  long ap, long b,int ldb );
        int dpptrint(int matriint, char uplo, int n,int nrhs,  long ap, long b,int ldb );
        int cpptrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                            IComplexNumber b, int ldb );
        int zpptrint(int matriint, char uplo, int n,int nrhs,
                            IComplexNumber ap,
                            IComplexNumber b, int ldb );

        int spstrint(int matriint, char uplo, int n,
                            long a, int lda, int piv,int rank, float tol, long work );
        int dpstrint(int matriint, char uplo, int n,
                            long a, int lda, int piv,int rank, double tol, long work );
        int cpstrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int piv, int rank, float tol,
                            long work );
        int zpstrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int piv, int rank, double tol,
                            long work );

        int sptcoint(int n,  long d,  long e,
                            float anorm, long rcond, long work );
        int dptcoint(int n,  long d,  long e,
                            double anorm, long rcond, long work );
        int cptcoint(int n,  long d,
                            IComplexNumber e, float anorm,
                            long rcond, long work );
        int zptcoint(int n,  long d,
                            IComplexNumber e, double anorm,
                            long rcond, long work );

        int spteqint(int matriint, char compz, int n,
                            long d, long e, long z, int ldz,
                            long work );
        int dpteqint(int matriint, char compz, int n,
                            long d, long e, long z, int ldz,
                            long work );
        int cpteqint(int matriint, char compz, int n,
                            long d, long e, IComplexNumber z,int ldz, long work );
        int zpteqint(int matriint, char compz, int n,
                            long d, long e, IComplexNumber z,int ldz, long work );

        int sptrfint(int matriint, int n, int nrhs,
                            long d,  long e,  long df,
                            long ef,  long b, int ldb,
                            long x, int ldx, long ferr,
                            long berr, long work );
        int dptrfint(int matriint, int n, int nrhs,
                            long d,  long e,
                            long df,  long ef,
                            long b, int ldb, long x,int ldx, long ferr, long berr,
                            long work );
        int cptrfint(int matriint, char uplo, int n,int nrhs,  long d,
                            IComplexNumber e,  long df,
                            IComplexNumber ef,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );
        int zptrfint(int matriint, char uplo, int n,int nrhs,  long d,
                            IComplexNumber e,
                            long df,
                            IComplexNumber ef,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );

        int sptsint(int matriint, int n, int nrhs,
                           long d, long e, long b, int ldb );
        int dptsint(int matriint, int n, int nrhs,
                           long d, long e, long b,int ldb );
        int cptsint(int matriint, int n, int nrhs,
                           long d, IComplexNumber e,
                           IComplexNumber b, int ldb );
        int zptsint(int matriint, int n, int nrhs,
                           long d, IComplexNumber e,
                           IComplexNumber b, int ldb );

        int sptsvint(int matriint, char fact, int n,int nrhs,  long d,  long e,
                            long df, long ef,  long b,int ldb, long x, int ldx,
                            long rcond, long ferr, long berr,
                            long work );
        int dptsvint(int matriint, char fact, int n,int nrhs,  long d,
                            long e, long df, long ef,
                            long b, int ldb, long x,int ldx, long rcond, long ferr,
                            long berr, long work );
        int cptsvint(int matriint, char fact, int n,int nrhs,  long d,
                            IComplexNumber e, long df,
                            IComplexNumber ef,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork );
        int zptsvint(int matriint, char fact, int n,int nrhs,  long d,
                            IComplexNumber e, long df,
                            IComplexNumber ef,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork );

        int spttrint(int n, long d, long e );
        int dpttrint(int n, long d, long e );
        int cpttrint(int n, long d,
                            IComplexNumber e );
        int zpttrint(int n, long d,
                            IComplexNumber e );

        int spttrint(int matriint, int n, int nrhs,
                            long d,  long e, long b,int ldb );
        int dpttrint(int matriint, int n, int nrhs,
                            long d,  long e, long b,int ldb );
        int cpttrint(int matriint, char uplo, int n,int nrhs,  long d,
                            IComplexNumber e,
                            IComplexNumber b, int ldb );
        int zpttrint(int matriint, char uplo, int n,int nrhs,  long d,
                            IComplexNumber e,
                            IComplexNumber b, int ldb );

        int ssbeint(int matriint, char jobz, char uplo,int n, int kd, long ab,int ldab, long w, long z,int ldz, long work );
        int dsbeint(int matriint, char jobz, char uplo,int n, int kd, long ab,int ldab, long w, long z,int ldz, long work );

        int ssbevint(int matriint, char jobz, char uplo,int n, int kd, long ab,int ldab, long w, long z,int ldz, long work, int lwork,int iwork, int liwork );
        int dsbevint(int matriint, char jobz, char uplo,int n, int kd, long ab,int ldab, long w, long z,int ldz, long work, int lwork,int iwork, int liwork );

        int ssbevint(int matriint, char jobz, char range,
                            char uplo, int n, int kd,
                            long ab, int ldab, long q,int ldq, float vl, float vu,int il, int iu, float abstol,int m, long w, long z,int ldz, long work, int iwork,int ifail );
        int dsbevint(int matriint, char jobz, char range,
                            char uplo, int n, int kd,
                            long ab, int ldab, long q,int ldq, double vl, double vu,int il, int iu, double abstol,int m, long w, long z,int ldz, long work, int iwork,int ifail );

        int ssbgsint(int matriint, char vect, char uplo,int n, int ka, int kb,
                            long ab, int ldab,  long bb,int ldbb, long x, int ldx,
                            long work );
        int dsbgsint(int matriint, char vect, char uplo,int n, int ka, int kb,
                            long ab, int ldab,  long bb,int ldbb, long x, int ldx,
                            long work );

        int ssbgint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                           long ab, int ldab, long bb,int ldbb, long w, long z,int ldz, long work );
        int dsbgint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                           long ab, int ldab, long bb,int ldbb, long w, long z,int ldz, long work );

        int ssbgvint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                            long ab, int ldab, long bb,int ldbb, long w, long z,int ldz, long work, int lwork,int iwork, int liwork );
        int dsbgvint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                            long ab, int ldab, long bb,int ldbb, long w, long z,int ldz, long work, int lwork,int iwork, int liwork );

        int ssbgvint(int matriint, char jobz, char range,
                            char uplo, int n, int ka,int kb, long ab, int ldab,
                            long bb, int ldbb, long q,int ldq, float vl, float vu,int il, int iu, float abstol,int m, long w, long z,int ldz, long work, int iwork,int ifail );
        int dsbgvint(int matriint, char jobz, char range,
                            char uplo, int n, int ka,int kb, long ab, int ldab,
                            long bb, int ldbb, long q,int ldq, double vl, double vu,int il, int iu, double abstol,int m, long w, long z,int ldz, long work, int iwork,int ifail );

        int ssbtrint(int matriint, char vect, char uplo,int n, int kd, long ab,int ldab, long d, long e, long q,int ldq, long work );
        int dsbtrint(int matriint, char vect, char uplo,int n, int kd, long ab,int ldab, long d, long e,
                            long q, int ldq, long work );

        int ssfrint(int matriint, char transr, char uplo,
                           char trans, int n, int k,
                           float alpha,  long a, int lda,
                           float beta, long c );
        int dsfrint(int matriint, char transr, char uplo,
                           char trans, int n, int k,
                           double alpha,  long a, int lda,
                           double beta, long c );

        int sspcoint(int matriint, char uplo, int n,
                            long ap,int ipiv,
                            float anorm, long rcond, long work,int iwork );
        int dspcoint(int matriint, char uplo, int n,
                            long ap,int ipiv,
                            double anorm, long rcond, long work,int iwork );
        int cspcoint(int matriint, char uplo, int n,
                            IComplexNumber ap,int ipiv, float anorm,
                            long rcond, IComplexNumber work );
        int zspcoint(int matriint, char uplo, int n,
                            IComplexNumber ap,int ipiv, double anorm,
                            long rcond, IComplexNumber work );

        int sspeint(int matriint, char jobz, char uplo,int n, long ap, long w, long z,int ldz, long work );
        int dspeint(int matriint, char jobz, char uplo,int n, long ap, long w, long z,int ldz, long work );

        int sspevint(int matriint, char jobz, char uplo,int n, long ap, long w, long z,int ldz, long work, int lwork,int iwork, int liwork );
        int dspevint(int matriint, char jobz, char uplo,int n, long ap, long w, long z,int ldz, long work, int lwork,int iwork, int liwork );

        int sspevint(int matriint, char jobz, char range,
                            char uplo, int n, long ap, float vl,
                            float vu, int il, int iu,
                            float abstol, int m, long w, long z,int ldz, long work, int iwork,int ifail );
        int dspevint(int matriint, char jobz, char range,
                            char uplo, int n, long ap, double vl,
                            double vu, int il, int iu,
                            double abstol, int m, long w,
                            long z, int ldz, long work,int iwork, int ifail );

        int sspgsint(int matriint, int itype, char uplo,int n, long ap,  long bp );
        int dspgsint(int matriint, int itype, char uplo,int n, long ap,  long bp );

        int sspgint(int matriint, int itype, char jobz,
                           char uplo, int n, long ap, long bp,
                           long w, long z, int ldz,
                           long work );
        int dspgint(int matriint, int itype, char jobz,
                           char uplo, int n, long ap, long bp,
                           long w, long z, int ldz,
                           long work );

        int sspgvint(int matriint, int itype, char jobz,
                            char uplo, int n, long ap, long bp,
                            long w, long z, int ldz, long work,int lwork, int iwork,int liwork );
        int dspgvint(int matriint, int itype, char jobz,
                            char uplo, int n, long ap, long bp,
                            long w, long z, int ldz,
                            long work, int lwork,int iwork, int liwork );

        int sspgvint(int matriint, int itype, char jobz,
                            char range, char uplo, int n, long ap,
                            long bp, float vl, float vu, int il,int iu, float abstol, int m,
                            long w, long z, int ldz, long work,int iwork, int ifail );
        int dspgvint(int matriint, int itype, char jobz,
                            char range, char uplo, int n, long ap,
                            long bp, double vl, double vu, int il,int iu, double abstol, int m,
                            long w, long z, int ldz,
                            long work, int iwork,int ifail );

        int ssprfint(int matriint, char uplo, int n,int nrhs,  long ap,
                            long afp,int ipiv,
                            long b, int ldb, long x,int ldx, long ferr, long berr,
                            long work, int iwork );
        int dsprfint(int matriint, char uplo, int n,int nrhs,  long ap,
                            long afp,int ipiv,
                            long b, int ldb, long x,int ldx, long ferr, long berr,
                            long work, int iwork );
        int csprfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                            IComplexNumber afp,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );
        int zsprfint(int matriint, char uplo, int n,int nrhs,
                            IComplexNumber ap,
                            IComplexNumber afp,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );

        int sspsint(int matriint, char uplo, int n,int nrhs, long ap, int ipiv,
                           long b, int ldb );
        int dspsint(int matriint, char uplo, int n,int nrhs, long ap, int ipiv,
                           long b, int ldb );
        int cspsint(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
        int zspsint(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );

        int sspsvint(int matriint, char fact, char uplo,int n, int nrhs,  long ap,
                            long afp, int ipiv,  long b,int ldb, long x, int ldx,
                            long rcond, long ferr, long berr,
                            long work, int iwork );
        int dspsvint(int matriint, char fact, char uplo,int n, int nrhs,  long ap,
                            long afp, int ipiv,  long b,int ldb, long x, int ldx,
                            long rcond, long ferr, long berr,
                            long work, int iwork );
        int cspsvint(int matriint, char fact, char uplo,int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber afp, int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork );
        int zspsvint(int matriint, char fact, char uplo,int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber afp, int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork );

        int ssptrint(int matriint, char uplo, int n,
                            long ap, long d, long e, long tau );
        int dsptrint(int matriint, char uplo, int n,
                            long ap, long d, long e, long tau );

        int ssptrint(int matriint, char uplo, int n,
                            long ap, int ipiv );
        int dsptrint(int matriint, char uplo, int n,
                            long ap, int ipiv );
        int csptrint(int matriint, char uplo, int n,
                            IComplexNumber ap, int ipiv );
        int zsptrint(int matriint, char uplo, int n,
                            IComplexNumber ap, int ipiv );

        int ssptrint(int matriint, char uplo, int n,
                            long ap,int ipiv,
                            long work );
        int dsptrint(int matriint, char uplo, int n,
                            long ap,int ipiv,
                            long work );
        int csptrint(int matriint, char uplo, int n,
                            IComplexNumber ap,int ipiv,
                            IComplexNumber work );
        int zsptrint(int matriint, char uplo, int n,
                            IComplexNumber ap,int ipiv,
                            IComplexNumber work );

        int ssptrint(int matriint, char uplo, int n,int nrhs,  long ap,int ipiv, long b,int ldb );
        int dsptrint(int matriint, char uplo, int n,int nrhs,  long ap,int ipiv, long b,int ldb );
        int csptrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
        int zsptrint(int matriint, char uplo, int n,int nrhs,
                            IComplexNumber ap,int ipiv,
                            IComplexNumber b, int ldb );

        int sstebint( char range, char order, int n, float vl,
                             float vu, int il, int iu,
                             float abstol,  long d,  long e,int m, int nsplit, long w,int iblock, int isplit,
                             long work, int iwork );
        int dstebint( char range, char order, int n, double vl,
                             double vu, int il, int iu,
                             double abstol,  long d,  long e,int m, int nsplit, long w,int iblock, int isplit,
                             long work, int iwork );

        int sstedint(int matriint, char compz, int n,
                            long d, long e, long z, int ldz,
                            long work, int lwork,int iwork, int liwork );
        int dstedint(int matriint, char compz, int n,
                            long d, long e, long z, int ldz,
                            long work, int lwork,int iwork, int liwork );
        int cstedint(int matriint, char compz, int n,
                            long d, long e, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );
        int zstedint(int matriint, char compz, int n,
                            long d, long e, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );

        int sstegint(int matriint, char jobz, char range,int n, long d, long e, float vl,
                            float vu, int il, int iu,
                            float abstol, int m, long w, long z,int ldz, int isuppz, long work,int lwork, int iwork,int liwork );
        int dstegint(int matriint, char jobz, char range,int n, long d, long e, double vl,
                            double vu, int il, int iu,
                            double abstol, int m, long w,
                            long z, int ldz, int isuppz,
                            long work, int lwork,int iwork, int liwork );
        int cstegint(int matriint, char jobz, char range,int n, long d, long e, float vl,
                            float vu, int il, int iu,
                            float abstol, int m, long w,
                            IComplexNumber z, int ldz,int isuppz, long work,int lwork, int iwork,int liwork );
        int zstegint(int matriint, char jobz, char range,int n, long d, long e, double vl,
                            double vu, int il, int iu,
                            double abstol, int m, long w,
                            IComplexNumber z, int ldz,int isuppz, long work,int lwork, int iwork,int liwork );

        int ssteiint(int matriint, int n,  long d,
                            long e, int m,  long w,int iblock,int isplit, long z,int ldz, long work, int iwork,int ifailv );
        int dsteiint(int matriint, int n,  long d,
                            long e, int m,  long w,int iblock,int isplit, long z,int ldz, long work, int iwork,int ifailv );
        int csteiint(int matriint, int n,  long d,
                            long e, int m,  long w,int iblock,int isplit,
                            IComplexNumber z, int ldz,
                            long work, int iwork,int ifailv );
        int zsteiint(int matriint, int n,  long d,
                            long e, int m,  long w,int iblock,int isplit,
                            IComplexNumber z, int ldz,
                            long work, int iwork,int ifailv );

        int sstemint(int matriint, char jobz, char range,int n, long d, long e, float vl,
                            float vu, int il, int iu,int m, long w, long z,int ldz, int nzc,int isuppz, int tryrac,
                            long work, int lwork,int iwork, int liwork );
        int dstemint(int matriint, char jobz, char range,int n, long d, long e, double vl,
                            double vu, int il, int iu,int m, long w, long z,int ldz, int nzc,int isuppz, int tryrac,
                            long work, int lwork,int iwork, int liwork );
        int cstemint(int matriint, char jobz, char range,int n, long d, long e, float vl,
                            float vu, int il, int iu,int m, long w,
                            IComplexNumber z, int ldz,int nzc, int isuppz,int tryrac, long work,int lwork, int iwork,int liwork );
        int zstemint(int matriint, char jobz, char range,int n, long d, long e, double vl,
                            double vu, int il, int iu,int m, long w,
                            IComplexNumber z, int ldz,int nzc, int isuppz,int tryrac, long work,int lwork, int iwork,int liwork );

        int ssteqint(int matriint, char compz, int n,
                            long d, long e, long z, int ldz,
                            long work );
        int dsteqint(int matriint, char compz, int n,
                            long d, long e, long z, int ldz,
                            long work );
        int csteqint(int matriint, char compz, int n,
                            long d, long e, IComplexNumber z,int ldz, long work );
        int zsteqint(int matriint, char compz, int n,
                            long d, long e, IComplexNumber z,int ldz, long work );

        int ssterint(int n, long d, long e );
        int dsterint(int n, long d, long e );

        int ssteint(int matriint, char jobz, int n,
                           long d, long e, long z, int ldz,
                           long work );
        int dsteint(int matriint, char jobz, int n,
                           long d, long e, long z, int ldz,
                           long work );

        int sstevint(int matriint, char jobz, int n,
                            long d, long e, long z, int ldz,
                            long work, int lwork,int iwork, int liwork );
        int dstevint(int matriint, char jobz, int n,
                            long d, long e, long z, int ldz,
                            long work, int lwork,int iwork, int liwork );

        int sstevint(int matriint, char jobz, char range,int n, long d, long e, float vl,
                            float vu, int il, int iu,
                            float abstol, int m, long w, long z,int ldz, int isuppz, long work,int lwork, int iwork,int liwork );
        int dstevint(int matriint, char jobz, char range,int n, long d, long e, double vl,
                            double vu, int il, int iu,
                            double abstol, int m, long w,
                            long z, int ldz, int isuppz,
                            long work, int lwork,int iwork, int liwork );

        int sstevint(int matriint, char jobz, char range,int n, long d, long e, float vl,
                            float vu, int il, int iu,
                            float abstol, int m, long w, long z,int ldz, long work, int iwork,int ifail );
        int dstevint(int matriint, char jobz, char range,int n, long d, long e, double vl,
                            double vu, int il, int iu,
                            double abstol, int m, long w,
                            long z, int ldz, long work,int iwork, int ifail );

        int ssycoint(int matriint, char uplo, int n,
                            long a, int lda,int ipiv, float anorm,
                            long rcond, long work, int iwork );
        int dsycoint(int matriint, char uplo, int n,
                            long a, int lda,int ipiv, double anorm,
                            long rcond, long work,int iwork );
        int csycoint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv, float anorm,
                            long rcond, IComplexNumber work );
        int zsycoint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv, double anorm,
                            long rcond, IComplexNumber work );

        int ssyequint(int matriint, char uplo, int n,
                             long a, int lda, long s,
                             long scond, long amax, long work );
        int dsyequint(int matriint, char uplo, int n,
                             long a, int lda, long s,
                             long scond, long amax, long work );
        int csyequint(int matriint, char uplo, int n,
                             IComplexNumber a, int lda,
                             long s, long scond, long amax,
                             IComplexNumber work );
        int zsyequint(int matriint, char uplo, int n,
                             IComplexNumber a, int lda,
                             long s, long scond, long amax,
                             IComplexNumber work );

        int ssyeint(int matriint, char jobz, char uplo,int n, long a, int lda, long w,
                           long work, int lwork );
        int dsyeint(int matriint, char jobz, char uplo,int n, long a, int lda,
                           long w, long work, int lwork );

        int ssyevint(int matriint, char jobz, char uplo,int n, long a, int lda,
                            long w, long work, int lwork,int iwork, int liwork );
        int dsyevint(int matriint, char jobz, char uplo,int n, long a, int lda,
                            long w, long work, int lwork,int iwork, int liwork );

        int ssyevint(int matriint, char jobz, char range,
                            char uplo, int n, long a,int lda, float vl, float vu,int il, int iu, float abstol,int m, long w, long z,int ldz, int isuppz, long work,int lwork, int iwork,int liwork );
        int dsyevint(int matriint, char jobz, char range,
                            char uplo, int n, long a,int lda, double vl, double vu,int il, int iu, double abstol,int m, long w, long z,int ldz, int isuppz,
                            long work, int lwork,int iwork, int liwork );

        int ssyevint(int matriint, char jobz, char range,
                            char uplo, int n, long a,int lda, float vl, float vu,int il, int iu, float abstol,int m, long w, long z,int ldz, long work, int lwork,int iwork, int ifail );
        int dsyevint(int matriint, char jobz, char range,
                            char uplo, int n, long a,int lda, double vl, double vu,int il, int iu, double abstol,int m, long w, long z,int ldz, long work, int lwork,int iwork, int ifail );

        int ssygsint(int matriint, int itype, char uplo,int n, long a, int lda,
                            long b, int ldb );
        int dsygsint(int matriint, int itype, char uplo,int n, long a, int lda,
                            long b, int ldb );

        int ssygint(int matriint, int itype, char jobz,
                           char uplo, int n, long a,int lda, long b, int ldb,
                           long w, long work, int lwork );
        int dsygint(int matriint, int itype, char jobz,
                           char uplo, int n, long a,int lda, long b, int ldb,
                           long w, long work, int lwork );

        int ssygvint(int matriint, int itype, char jobz,
                            char uplo, int n, long a,int lda, long b, int ldb,
                            long w, long work, int lwork,int iwork, int liwork );
        int dsygvint(int matriint, int itype, char jobz,
                            char uplo, int n, long a,int lda, long b, int ldb,
                            long w, long work, int lwork,int iwork, int liwork );

        int ssygvint(int matriint, int itype, char jobz,
                            char range, char uplo, int n, long a,int lda, long b, int ldb,
                            float vl, float vu, int il,int iu, float abstol, int m,
                            long w, long z, int ldz, long work,int lwork, int iwork,int ifail );
        int dsygvint(int matriint, int itype, char jobz,
                            char range, char uplo, int n, long a,int lda, long b, int ldb,
                            double vl, double vu, int il,int iu, double abstol, int m,
                            long w, long z, int ldz,
                            long work, int lwork,int iwork, int ifail );

        int ssyrfint(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                            long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                            long ferr, long berr, long work,int iwork );
        int dsyrfint(int matriint, char uplo, int n,int nrhs,  long a,int lda,  long af,int ldaf,int ipiv,
                            long b, int ldb, long x,int ldx, long ferr, long berr,
                            long work, int iwork );
        int csyrfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );
        int zsyrfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );

        int ssyrfsint(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af,int ldaf,int ipiv,
                             long s,  long b, int ldb,
                             long x, int ldx, long rcond,
                             long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params, long work,int iwork );
        int dsyrfsint(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af,int ldaf,int ipiv,
                             long s,  long b,int ldb, long x, int ldx,
                             long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, long work,int iwork );
        int csyrfsint(int matriint, char uplo, char equed,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af,int ldaf,int ipiv,
                             long s,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );
        int zsyrfsint(int matriint, char uplo, char equed,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af,int ldaf,int ipiv,
                             long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx,
                             long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );

        int ssysint(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb,
                           long work, int lwork );
        int dsysint(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb,
                           long work, int lwork );
        int csysint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                           IComplexNumber b, int ldb,
                           IComplexNumber work, int lwork );
        int zsysint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                           IComplexNumber b, int ldb,
                           IComplexNumber work, int lwork );

        int ssysvint(int matriint, char fact, char uplo,int n, int nrhs,  long a,int lda, long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                            long rcond, long ferr, long berr,
                            long work, int lwork,int iwork );
        int dsysvint(int matriint, char fact, char uplo,int n, int nrhs,  long a,int lda, long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                            long rcond, long ferr, long berr,
                            long work, int lwork,int iwork );
        int csysvint(int matriint, char fact, char uplo,int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                            long berr, IComplexNumber work,int lwork, long rwork );
        int zsysvint(int matriint, char fact, char uplo,int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, int lwork,
                            long rwork );

        int ssysvxint(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long s,
                             long b, int ldb, long x,int ldx, long rcond, long rpvgrw,
                             long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params, long work,int iwork );
        int dsysvxint(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long s,
                             long b, int ldb, long x,int ldx, long rcond, long rpvgrw,
                             long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             long work, int iwork );
        int csysvxint(int matriint, char fact, char uplo,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx,
                             long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );
        int zsysvxint(int matriint, char fact, char uplo,int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx,
                             long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams,
                             long params, IComplexNumber work,
                             long rwork );

        int ssytrint(int matriint, char uplo, int n,
                            long a, int lda, long d, long e,
                            long tau, long work, int lwork );
        int dsytrint(int matriint, char uplo, int n,
                            long a, int lda, long d, long e,
                            long tau, long work, int lwork );

        int ssytrint(int matriint, char uplo, int n,
                            long a, int lda, int ipiv,
                            long work, int lwork );
        int dsytrint(int matriint, char uplo, int n,
                            long a, int lda, int ipiv,
                            long work, int lwork );
        int csytrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );
        int zsytrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );

        int ssytrint(int matriint, char uplo, int n,
                            long a, int lda,int ipiv, long work );
        int dsytrint(int matriint, char uplo, int n,
                            long a, int lda,int ipiv, long work );
        int csytrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv,
                            IComplexNumber work );
        int zsytrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv,
                            IComplexNumber work );



        int stbcoint(int matriint, char norm, char uplo,
                            char diag, int n, int kd,
                            long ab, int ldab, long rcond,
                            long work, int iwork );
        int dtbcoint(int matriint, char norm, char uplo,
                            char diag, int n, int kd,
                            long ab, int ldab,
                            long rcond, long work,int iwork );
        int ctbcoint(int matriint, char norm, char uplo,
                            char diag, int n, int kd,
                            IComplexNumber ab, int ldab,
                            long rcond, IComplexNumber work,
                            long rwork );
        int ztbcoint(int matriint, char norm, char uplo,
                            char diag, int n, int kd,
                            IComplexNumber ab,int ldab, long rcond,
                            IComplexNumber work, long rwork );

        int stbrfint(int matriint, char uplo, char trans,
                            char diag, int n, int kd,int nrhs,  long ab,int ldab,  long b, int ldb,
                            long x, int ldx, long ferr,
                            long berr, long work, int iwork );
        int dtbrfint(int matriint, char uplo, char trans,
                            char diag, int n, int kd,int nrhs,  long ab,int ldab,  long b,int ldb,  long x, int ldx,
                            long ferr, long berr, long work,int iwork );
        int ctbrfint(int matriint, char uplo, char trans,
                            char diag, int n, int kd,int nrhs,  IComplexNumber ab,int ldab,  IComplexNumber b,int ldb,  IComplexNumber x,int ldx, long ferr, long berr,
                            IComplexNumber work, long rwork );
        int ztbrfint(int matriint, char uplo, char trans,
                            char diag, int n, int kd,int nrhs,
                            IComplexNumber ab,int ldab,  IComplexNumber b,int ldb,  IComplexNumber x,int ldx, long ferr, long berr,
                            IComplexNumber work, long rwork );

        int stbtrint(int matriint, char uplo, char trans,
                            char diag, int n, int kd,int nrhs,  long ab,int ldab, long b, int ldb );
        int dtbtrint(int matriint, char uplo, char trans,
                            char diag, int n, int kd,int nrhs,  long ab,int ldab, long b, int ldb );
        int ctbtrint(int matriint, char uplo, char trans,
                            char diag, int n, int kd,int nrhs,  IComplexNumber ab,int ldab, IComplexNumber b,int ldb );
        int ztbtrint(int matriint, char uplo, char trans,
                            char diag, int n, int kd,int nrhs,
                            IComplexNumber ab,int ldab, IComplexNumber b,int ldb );

        int stfsint(int matriint, char transr, char side,
                           char uplo, char trans, char diag, int m,int n, float alpha,  long a,
                           long b, int ldb );
        int dtfsint(int matriint, char transr, char side,
                           char uplo, char trans, char diag, int m,int n, double alpha,  long a,
                           long b, int ldb );
        int ctfsint(int matriint, char transr, char side,
                           char uplo, char trans, char diag, int m,int n, IComplexNumber alpha,
                           IComplexNumber a,
                           IComplexNumber b, int ldb );
        int ztfsint(int matriint, char transr, char side,
                           char uplo, char trans, char diag, int m,int n, IComplexNumber alpha,
                           IComplexNumber a,
                           IComplexNumber b, int ldb );

        int stftrint(int matriint, char transr, char uplo,
                            char diag, int n, long a );
        int dtftrint(int matriint, char transr, char uplo,
                            char diag, int n, long a );
        int ctftrint(int matriint, char transr, char uplo,
                            char diag, int n,
                            IComplexNumber a );
        int ztftrint(int matriint, char transr, char uplo,
                            char diag, int n,
                            IComplexNumber a );

        int stfttint(int matriint, char transr, char uplo,int n,  long arf, long ap );
        int dtfttint(int matriint, char transr, char uplo,int n,  long arf, long ap );
        int ctfttint(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                            IComplexNumber ap );
        int ztfttint(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                            IComplexNumber ap );

        int stfttint(int matriint, char transr, char uplo,int n,  long arf, long a,int lda );
        int dtfttint(int matriint, char transr, char uplo,int n,  long arf, long a,int lda );
        int ctfttint(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                            IComplexNumber a, int lda );
        int ztfttint(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                            IComplexNumber a, int lda );

        int stgevint(int matriint, char side, char howmny,int select, int n,
                            long s, int lds,  long p,int ldp, long vl, int ldvl,
                            long vr, int ldvr, int mm,int m, long work );
        int dtgevint(int matriint, char side, char howmny,int select, int n,
                            long s, int lds,
                            long p, int ldp, long vl,int ldvl, long vr, int ldvr,int mm, int m, long work );
        int ctgevint(int matriint, char side, char howmny,int select, int n,
                            IComplexNumber s, int lds,
                            IComplexNumber p, int ldp,
                            IComplexNumber vl, int ldvl,
                            IComplexNumber vr, int ldvr,int mm, int m,
                            IComplexNumber work, long rwork );
        int ztgevint(int matriint, char side, char howmny,int select, int n,
                            IComplexNumber s, int lds,
                            IComplexNumber p, int ldp,
                            IComplexNumber vl, int ldvl,
                            IComplexNumber vr, int ldvr,int mm, int m,
                            IComplexNumber work, long rwork );

        int stgexint(int matriint, int wantq,int wantz, int n, long a,int lda, long b, int ldb,
                            long q, int ldq, long z,int ldz, int ifst,int ilst, long work,int lwork );
        int dtgexint(int matriint, int wantq,int wantz, int n, long a,int lda, long b, int ldb,
                            long q, int ldq, long z,int ldz, int ifst,int ilst, long work,int lwork );
        int ctgexint(int matriint, int wantq,int wantz, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber q, int ldq,
                            IComplexNumber z, int ldz,int ifst, int ilst );
        int ztgexint(int matriint, int wantq,int wantz, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber q, int ldq,
                            IComplexNumber z, int ldz,int ifst, int ilst );

        int stgseint(int matriint, int ijob,int wantq, int wantz,int select, int n,
                            long a, int lda, long b,int ldb, long alphar, long alphai,
                            long beta, long q, int ldq, long z,int ldz, int m, long pl,
                            long pr, long dif, long work,int lwork, int iwork,int liwork );
        int dtgseint(int matriint, int ijob,int wantq, int wantz,int select, int n,
                            long a, int lda, long b,int ldb, long alphar, long alphai,
                            long beta, long q, int ldq,
                            long z, int ldz, int m,
                            long pl, long pr, long dif,
                            long work, int lwork,int iwork, int liwork );
        int ctgseint(int matriint, int ijob,int wantq, int wantz,int select, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber alpha,
                            IComplexNumber beta,
                            IComplexNumber q, int ldq,
                            IComplexNumber z, int ldz,int m, long pl, long pr, long dif,
                            IComplexNumber work, int lwork,int iwork, int liwork );
        int ztgseint(int matriint, int ijob,int wantq, int wantz,int select, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber alpha,
                            IComplexNumber beta,
                            IComplexNumber q, int ldq,
                            IComplexNumber z, int ldz,int m, long pl, long pr,
                            long dif, IComplexNumber work,int lwork, int iwork,int liwork );

        int stgsjint(int matriint, char jobu, char jobv,
                            char jobq, int m, int p,int n, int k, int l,
                            long a, int lda, long b,int ldb, float tola, float tolb,
                            long alpha, long beta, long u,int ldu, long v, int ldv,
                            long q, int ldq, long work,int ncycle );
        int dtgsjint(int matriint, char jobu, char jobv,
                            char jobq, int m, int p,int n, int k, int l,
                            long a, int lda, long b,int ldb, double tola, double tolb,
                            long alpha, long beta, long u,int ldu, long v, int ldv,
                            long q, int ldq, long work,int ncycle );
        int ctgsjint(int matriint, char jobu, char jobv,
                            char jobq, int m, int p,int n, int k, int l,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            float tola, float tolb, long alpha,
                            long beta, IComplexNumber u,int ldu, IComplexNumber v,int ldv, IComplexNumber q,int ldq, IComplexNumber work,int ncycle );
        int ztgsjint(int matriint, char jobu, char jobv,
                            char jobq, int m, int p,int n, int k, int l,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            double tola, double tolb, long alpha,
                            long beta, IComplexNumber u,int ldu, IComplexNumber v,int ldv, IComplexNumber q,int ldq, IComplexNumber work,int ncycle );

        int stgsnint(int matriint, char job, char howmny,int select, int n,
                            long a, int lda,  long b,int ldb,  long vl,int ldvl,  long vr,int ldvr, long s, long dif,int mm, int m, long work,int lwork, int iwork );
        int dtgsnint(int matriint, char job, char howmny,int select, int n,
                            long a, int lda,
                            long b, int ldb,
                            long vl, int ldvl,
                            long vr, int ldvr, long s,
                            long dif, int mm, int m,
                            long work, int lwork,int iwork );
        int ctgsnint(int matriint, char job, char howmny,int select, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber vl, int ldvl,
                            IComplexNumber vr, int ldvr,
                            long s, long dif, int mm,int m, IComplexNumber work,int lwork, int iwork );
        int ztgsnint(int matriint, char job, char howmny,int select, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber vl,int ldvl,
                            IComplexNumber vr,int ldvr, long s, long dif,int mm, int m,
                            IComplexNumber work, int lwork,int iwork );

        int stgsyint(int matriint, char trans, int ijob,int m, int n,  long a,int lda,  long b, int ldb,
                            long c, int ldc,  long d,int ldd,  long e, int lde,
                            long f, int ldf, long scale,
                            long dif, long work, int lwork,int iwork );
        int dtgsyint(int matriint, char trans, int ijob,int m, int n,  long a,int lda,  long b, int ldb,
                            long c, int ldc,  long d,int ldd,  long e, int lde,
                            long f, int ldf, long scale,
                            long dif, long work, int lwork,int iwork );
        int ctgsyint(int matriint, char trans, int ijob,int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber c, int ldc,
                            IComplexNumber d, int ldd,
                            IComplexNumber e, int lde,
                            IComplexNumber f, int ldf,
                            long scale, long dif,
                            IComplexNumber work, int lwork,int iwork );
        int ztgsyint(int matriint, char trans, int ijob,int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber c, int ldc,
                            IComplexNumber d, int ldd,
                            IComplexNumber e, int lde,
                            IComplexNumber f, int ldf,
                            long scale, long dif,
                            IComplexNumber work, int lwork,int iwork );

        int stpcoint(int matriint, char norm, char uplo,
                            char diag, int n,  long ap,
                            long rcond, long work, int iwork );
        int dtpcoint(int matriint, char norm, char uplo,
                            char diag, int n,  long ap,
                            long rcond, long work,int iwork );
        int ctpcoint(int matriint, char norm, char uplo,
                            char diag, int n,
                            IComplexNumber ap, long rcond,
                            IComplexNumber work, long rwork );
        int ztpcoint(int matriint, char norm, char uplo,
                            char diag, int n,
                            IComplexNumber ap, long rcond,
                            IComplexNumber work, long rwork );

        int stprfint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            long ap,  long b, int ldb,
                            long x, int ldx, long ferr,
                            long berr, long work, int iwork );
        int dtprfint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            long ap,  long b,int ldb,  long x, int ldx,
                            long ferr, long berr, long work,int iwork );
        int ctprfint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );
        int ztprfint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );

        int stptrint(int matriint, char uplo, char diag,int n, long ap );
        int dtptrint(int matriint, char uplo, char diag,int n, long ap );
        int ctptrint(int matriint, char uplo, char diag,int n, IComplexNumber ap );
        int ztptrint(int matriint, char uplo, char diag,int n, IComplexNumber ap );

        int stptrint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            long ap, long b, int ldb );
        int dtptrint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            long ap, long b, int ldb );
        int ctptrint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber b, int ldb );
        int ztptrint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber b, int ldb );

        int stpttint(int matriint, char transr, char uplo,int n,  long ap, long arf );
        int dtpttint(int matriint, char transr, char uplo,int n,  long ap, long arf );
        int ctpttint(int matriint, char transr, char uplo,int n,  IComplexNumber ap,
                            IComplexNumber arf );
        int ztpttint(int matriint, char transr, char uplo,int n,  IComplexNumber ap,
                            IComplexNumber arf );

        int stpttint(int matriint, char uplo, int n,
                            long ap, long a, int lda );
        int dtpttint(int matriint, char uplo, int n,
                            long ap, long a, int lda );
        int ctpttint(int matriint, char uplo, int n,
                            IComplexNumber ap,
                            IComplexNumber a, int lda );
        int ztpttint(int matriint, char uplo, int n,
                            IComplexNumber ap,
                            IComplexNumber a, int lda );

        int strcoint(int matriint, char norm, char uplo,
                            char diag, int n,  long a,int lda, long rcond, long work,int iwork );
        int dtrcoint(int matriint, char norm, char uplo,
                            char diag, int n,  long a,int lda, long rcond, long work,int iwork );
        int ctrcoint(int matriint, char norm, char uplo,
                            char diag, int n,
                            IComplexNumber a, int lda,
                            long rcond, IComplexNumber work,
                            long rwork );
        int ztrcoint(int matriint, char norm, char uplo,
                            char diag, int n,
                            IComplexNumber a, int lda,
                            long rcond, IComplexNumber work,
                            long rwork );

        int strevint(int matriint, char side, char howmny,int select, int n,
                            long t, int ldt, long vl,int ldvl, long vr, int ldvr,int mm, int m, long work );
        int dtrevint(int matriint, char side, char howmny,int select, int n,
                            long t, int ldt, long vl,int ldvl, long vr, int ldvr,int mm, int m, long work );
        int ctrevint(int matriint, char side, char howmny,int select, int n,
                            IComplexNumber t, int ldt,
                            IComplexNumber vl, int ldvl,
                            IComplexNumber vr, int ldvr,int mm, int m,
                            IComplexNumber work, long rwork );
        int ztrevint(int matriint, char side, char howmny,int select, int n,
                            IComplexNumber t, int ldt,
                            IComplexNumber vl, int ldvl,
                            IComplexNumber vr, int ldvr,int mm, int m,
                            IComplexNumber work, long rwork );

        int strexint(int matriint, char compq, int n,
                            long t, int ldt, long q,int ldq, int ifst,int ilst, long work );
        int dtrexint(int matriint, char compq, int n,
                            long t, int ldt, long q,int ldq, int ifst,int ilst, long work );
        int ctrexint(int matriint, char compq, int n,
                            IComplexNumber t, int ldt,
                            IComplexNumber q, int ldq,int ifst, int ilst );
        int ztrexint(int matriint, char compq, int n,
                            IComplexNumber t, int ldt,
                            IComplexNumber q, int ldq,int ifst, int ilst );

        int strrfint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            long a, int lda,  long b,int ldb,  long x, int ldx,
                            long ferr, long berr, long work,int iwork );
        int dtrrfint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            long a, int lda,
                            long b, int ldb,
                            long x, int ldx, long ferr,
                            long berr, long work, int iwork );
        int ctrrfint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );
        int ztrrfint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr,
                            IComplexNumber work, long rwork );

        int strseint(int matriint, char job, char compq,int select, int n,
                            long t, int ldt, long q,int ldq, long wr, long wi,int m, long s, long sep,
                            long work, int lwork,int iwork, int liwork );
        int dtrseint(int matriint, char job, char compq,int select, int n,
                            long t, int ldt, long q,int ldq, long wr, long wi,int m, long s, long sep,
                            long work, int lwork,int iwork, int liwork );
        int ctrseint(int matriint, char job, char compq,int select, int n,
                            IComplexNumber t, int ldt,
                            IComplexNumber q, int ldq,
                            IComplexNumber w, int m,
                            long s, long sep,
                            IComplexNumber work, int lwork );
        int ztrseint(int matriint, char job, char compq,int select, int n,
                            IComplexNumber t, int ldt,
                            IComplexNumber q, int ldq,
                            IComplexNumber w, int m,
                            long s, long sep,
                            IComplexNumber work, int lwork );

        int strsnint(int matriint, char job, char howmny,int select, int n,
                            long t, int ldt,  long vl,int ldvl,  long vr,int ldvr, long s, long sep,int mm, int m, long work,int ldwork, int iwork );
        int dtrsnint(int matriint, char job, char howmny,int select, int n,
                            long t, int ldt,
                            long vl, int ldvl,
                            long vr, int ldvr, long s,
                            long sep, int mm, int m,
                            long work, int ldwork,int iwork );
        int ctrsnint(int matriint, char job, char howmny,int select, int n,
                            IComplexNumber t, int ldt,
                            IComplexNumber vl, int ldvl,
                            IComplexNumber vr, int ldvr,
                            long s, long sep, int mm,int m, IComplexNumber work,int ldwork, long rwork );
        int ztrsnint(int matriint, char job, char howmny,int select, int n,
                            IComplexNumber t, int ldt,
                            IComplexNumber vl,int ldvl,
                            IComplexNumber vr,int ldvr, long s, long sep,int mm, int m,
                            IComplexNumber work, int ldwork,
                            long rwork );

        int strsyint(int matriint, char trana, char tranb,int isgn, int m, int n,
                            long a, int lda,  long b,int ldb, long c, int ldc,
                            long scale );
        int dtrsyint(int matriint, char trana, char tranb,int isgn, int m, int n,
                            long a, int lda,
                            long b, int ldb, long c,int ldc, long scale );
        int ctrsyint(int matriint, char trana, char tranb,int isgn, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber c, int ldc,
                            long scale );
        int ztrsyint(int matriint, char trana, char tranb,int isgn, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber c, int ldc,
                            long scale );

        int strtrint(int matriint, char uplo, char diag,int n, long a, int lda );
        int dtrtrint(int matriint, char uplo, char diag,int n, long a, int lda );
        int ctrtrint(int matriint, char uplo, char diag,int n, IComplexNumber a,int lda );
        int ztrtrint(int matriint, char uplo, char diag,int n, IComplexNumber a,int lda );

        int strtrint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            long a, int lda, long b,int ldb );
        int dtrtrint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            long a, int lda, long b,int ldb );
        int ctrtrint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb );
        int ztrtrint(int matriint, char uplo, char trans,
                            char diag, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb );

        int strttint(int matriint, char transr, char uplo,int n,  long a, int lda,
                            long arf );
        int dtrttint(int matriint, char transr, char uplo,int n,  long a, int lda,
                            long arf );
        int ctrttint(int matriint, char transr, char uplo,int n,  IComplexNumber a,int lda, IComplexNumber arf );
        int ztrttint(int matriint, char transr, char uplo,int n,  IComplexNumber a,int lda, IComplexNumber arf );

        int strttint(int matriint, char uplo, int n,
                            long a, int lda, long ap );
        int dtrttint(int matriint, char uplo, int n,
                            long a, int lda, long ap );
        int ctrttint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber ap );
        int ztrttint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber ap );

        int stzrzint(int matriint, int m, int n,
                            long a, int lda, long tau,
                            long work, int lwork );
        int dtzrzint(int matriint, int m, int n,
                            long a, int lda, long tau,
                            long work, int lwork );
        int ctzrzint(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );
        int ztzrzint(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );

        int cungbint(int matriint, char vect, int m,int n, int k,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );
        int zungbint(int matriint, char vect, int m,int n, int k,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );

        int cunghint(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber work, int lwork );
        int zunghint(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );

        int cunglint(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber work, int lwork );
        int zunglint(int matriint, int m, int n,int k, IComplexNumber a,int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );

        int cungqint(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber work, int lwork );
        int zungqint(int matriint, int m, int n,int k, IComplexNumber a,int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );


        int cungrint(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber work, int lwork );
        int zungrint(int matriint, int m, int n,int k, IComplexNumber a,int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );

        int cungtint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );
        int zungtint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );

        int cunmbint(int matriint, char vect, char side,
                            char trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork );
        int zunmbint(int matriint, char vect, char side,
                            char trans, int m, int n,int k,  IComplexNumber a,int lda,
                            IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork );

        int cunmhint(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork );
        int zunmhint(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  IComplexNumber a,int lda,
                            IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork );

        int cunmlint(int matriint, char side, char trans,int m, int n, int k,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork );
        int zunmlint(int matriint, char side, char trans,int m, int n, int k,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork );

        int cunmqint(int matriint, char side, char trans,int m, int n, int k,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork );
        int zunmqint(int matriint, char side, char trans,int m, int n, int k,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork );



        int cunmrint(int matriint, char side, char trans,int m, int n, int k,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork );
        int zunmrint(int matriint, char side, char trans,int m, int n, int k,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork );

        int cunmrint(int matriint, char side, char trans,int m, int n, int k,int l,  IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork );
        int zunmrint(int matriint, char side, char trans,int m, int n, int k,int l,  IComplexNumber a,int lda,
                            IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork );

        int cunmtint(int matriint, char side, char uplo,
                            char trans, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork );
        int zunmtint(int matriint, char side, char uplo,
                            char trans, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork );

        int cupgtint(int matriint, char uplo, int n,
                            IComplexNumber ap,
                            IComplexNumber tau,
                            IComplexNumber q, int ldq,
                            IComplexNumber work );
        int zupgtint(int matriint, char uplo, int n,
                            IComplexNumber ap,
                            IComplexNumber tau,
                            IComplexNumber q, int ldq,
                            IComplexNumber work );

        int cupmtint(int matriint, char side, char uplo,
                            char trans, int m, int n,
                            IComplexNumber ap,
                            IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work );
        int zupmtint(int matriint, char side, char uplo,
                            char trans, int m, int n,
                            IComplexNumber ap,
                            IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work );

        int claghe(int matriint, int n, int k,
                          long d, IComplexNumber a,int lda, int iseed );
        int zlaghe(int matriint, int n, int k,
                          long d, IComplexNumber a,int lda, int iseed );

        int slagsy(int matriint, int n, int k,
                          long d, long a, int lda,int iseed );
        int dlagsy(int matriint, int n, int k,
                          long d, long a, int lda,int iseed );
        int clagsy(int matriint, int n, int k,
                          long d, IComplexNumber a,int lda, int iseed );
        int zlagsy(int matriint, int n, int k,
                          long d, IComplexNumber a,int lda, int iseed );

        int slapmr(int matriint, int forwrd,int m, int n, long x, int ldx,int k );
        int dlapmr(int matriint, int forwrd,int m, int n, long x,int ldx, int k );
        int clapmr(int matriint, int forwrd,int m, int n, IComplexNumber x,int ldx, int k );
        int zlapmr(int matriint, int forwrd,int m, int n, IComplexNumber x,int ldx, int k );

        int slapmt(int matriint, int forwrd,int m, int n, long x, int ldx,int k );
        int dlapmt(int matriint, int forwrd,int m, int n, long x,int ldx, int k );
        int clapmt(int matriint, int forwrd,int m, int n, IComplexNumber x,int ldx, int k );
        int zlapmt(int matriint, int forwrd,int m, int n, IComplexNumber x,int ldx, int k );

        float slapy2( float x, float y );
        double dlapy2( double x, double y );

        float slapy3( float x, float y, float z );
        double dlapy3( double x, double y, double z );

        int slartgp( float f, float g, long cs, long sn, long r );
        int dlartgp( double f, double g, long cs, long sn,
                            long r );

        int slartgs( float x, float y, float sigma, long cs,
                            long sn );
        int dlartgs( double x, double y, double sigma, long cs,
                            long sn );


//LAPACK 3.3.0
        int cbbcsd(int matriint, char jobu1, char jobu2,
                          char jobv1t, char jobv2t, char trans, int m,int p, int q, long theta, long phi,
                          IComplexNumber u1, int ldu1,
                          IComplexNumber u2, int ldu2,
                          IComplexNumber v1t, int ldv1t,
                          IComplexNumber v2t, int ldv2t,
                          long b11d, long b11e, long b12d, long b12e,
                          long b21d, long b21e, long b22d, long b22e );
        int cbbcsint(int matriint, char jobu1, char jobu2,
                            char jobv1t, char jobv2t, char trans,int m, int p, int q,
                            long theta, long phi,
                            IComplexNumber u1, int ldu1,
                            IComplexNumber u2, int ldu2,
                            IComplexNumber v1t, int ldv1t,
                            IComplexNumber v2t, int ldv2t,
                            long b11d, long b11e, long b12d,
                            long b12e, long b21d, long b21e,
                            long b22d, long b22e, long rwork,int lrwork );
        int cheswapr(int matriint, char uplo, int n,
                            IComplexNumber a, int i1,int i2 );
        int cheswapint(int matriint, char uplo, int n,
                              IComplexNumber a, int i1,int i2 );
        int chetri2(int matriint, char uplo, int n,
                           IComplexNumber a, int lda,int ipiv );
        int chetri2_work(int matriint, char uplo, int n,
                                IComplexNumber a, int lda,int ipiv,
                                IComplexNumber work, int lwork );
        int chetri2x(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv, int nb );
        int chetri2int(int matriint, char uplo, int n,
                              IComplexNumber a, int lda,int ipiv,
                              IComplexNumber work, int nb );
        int chetrs2(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                           IComplexNumber b, int ldb );
        int chetrs2_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                                IComplexNumber b, int ldb,
                                IComplexNumber work );
        int csyconv(int matriint, char uplo, char way, int n,
                           IComplexNumber a, int lda,int ipiv, IComplexNumber work  );
        int csyconint(int matriint, char uplo, char way,int n, IComplexNumber a,int lda,int ipiv,
                             IComplexNumber work );
        int csyswapr(int matriint, char uplo, int n,
                            IComplexNumber a, int i1,int i2 );
        int csyswapint(int matriint, char uplo, int n,
                              IComplexNumber a, int i1,int i2 );
        int csytri2(int matriint, char uplo, int n,
                           IComplexNumber a, int lda,int ipiv );
        int csytri2_work(int matriint, char uplo, int n,
                                IComplexNumber a, int lda,int ipiv,
                                IComplexNumber work, int lwork );
        int csytri2x(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv, int nb );
        int csytri2int(int matriint, char uplo, int n,
                              IComplexNumber a, int lda,int ipiv,
                              IComplexNumber work, int nb );
        int csytrs2(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                           IComplexNumber b, int ldb );
        int csytrs2_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                                IComplexNumber b, int ldb,
                                IComplexNumber work );
        int cunbdb(int matriint, char trans, char signs,int m, int p, int q,
                          IComplexNumber x11, int ldx11,
                          IComplexNumber x12, int ldx12,
                          IComplexNumber x21, int ldx21,
                          IComplexNumber x22, int ldx22,
                          long theta, long phi,
                          IComplexNumber taup1,
                          IComplexNumber taup2,
                          IComplexNumber tauq1,
                          IComplexNumber tauq2 );
        int cunbdint(int matriint, char trans, char signs,int m, int p, int q,
                            IComplexNumber x11, int ldx11,
                            IComplexNumber x12, int ldx12,
                            IComplexNumber x21, int ldx21,
                            IComplexNumber x22, int ldx22,
                            long theta, long phi,
                            IComplexNumber taup1,
                            IComplexNumber taup2,
                            IComplexNumber tauq1,
                            IComplexNumber tauq2,
                            IComplexNumber work, int lwork );
        int cuncsd(int matriint, char jobu1, char jobu2,
                          char jobv1t, char jobv2t, char trans, char signs,int m, int p, int q,
                          IComplexNumber x11, int ldx11,
                          IComplexNumber x12, int ldx12,
                          IComplexNumber x21, int ldx21,
                          IComplexNumber x22, int ldx22,
                          long theta, IComplexNumber u1,int ldu1, IComplexNumber u2,int ldu2, IComplexNumber v1t,int ldv1t, IComplexNumber v2t,int ldv2t );
        int cuncsint(int matriint, char jobu1, char jobu2,
                            char jobv1t, char jobv2t, char trans,
                            char signs, int m, int p,int q, IComplexNumber x11,int ldx11, IComplexNumber x12,int ldx12, IComplexNumber x21,int ldx21, IComplexNumber x22,int ldx22, long theta,
                            IComplexNumber u1, int ldu1,
                            IComplexNumber u2, int ldu2,
                            IComplexNumber v1t, int ldv1t,
                            IComplexNumber v2t, int ldv2t,
                            IComplexNumber work, int lwork,
                            long rwork, int lrwork,int iwork );
        int cuncsd2by1(int matriint, char jobu1, char jobu2,
                              char jobv1t, int m, int p, int q,
                              IComplexNumber x11, int ldx11,
                              IComplexNumber x21, int ldx21,
                              IComplexNumber theta, IComplexNumber u1,int ldu1, IComplexNumber u2,int ldu2, IComplexNumber v1t, int ldv1t );
        int cuncsd2by1_work(int matriint, char jobu1, char jobu2,
                                   char jobv1t, int m, int p,int q, IComplexNumber x11, int ldx11,
                                   IComplexNumber x21, int ldx21,
                                   IComplexNumber theta, IComplexNumber u1,int ldu1, IComplexNumber u2,int ldu2, IComplexNumber v1t,int ldv1t, IComplexNumber work,int lwork, long rwork, int lrwork,int iwork );
        int dbbcsd(int matriint, char jobu1, char jobu2,
                          char jobv1t, char jobv2t, char trans, int m,int p, int q, long theta,
                          long phi, long u1, int ldu1, long u2,int ldu2, long v1t, int ldv1t,
                          long v2t, int ldv2t, long b11d,
                          long b11e, long b12d, long b12e,
                          long b21d, long b21e, long b22d,
                          long b22e );
        int dbbcsint(int matriint, char jobu1, char jobu2,
                            char jobv1t, char jobv2t, char trans,int m, int p, int q,
                            long theta, long phi, long u1,int ldu1, long u2, int ldu2,
                            long v1t, int ldv1t, long v2t,int ldv2t, long b11d, long b11e,
                            long b12d, long b12e, long b21d,
                            long b21e, long b22d, long b22e,
                            long work, int lwork );
        int dorbdb(int matriint, char trans, char signs,int m, int p, int q,
                          long x11, int ldx11, long x12,int ldx12, long x21, int ldx21,
                          long x22, int ldx22, long theta,
                          long phi, long taup1, long taup2,
                          long tauq1, long tauq2 );
        int dorbdint(int matriint, char trans, char signs,int m, int p, int q,
                            long x11, int ldx11, long x12,int ldx12, long x21, int ldx21,
                            long x22, int ldx22, long theta,
                            long phi, long taup1, long taup2,
                            long tauq1, long tauq2, long work,int lwork );
        int dorcsd(int matriint, char jobu1, char jobu2,
                          char jobv1t, char jobv2t, char trans, char signs,int m, int p, int q,
                          long x11, int ldx11, long x12,int ldx12, long x21, int ldx21,
                          long x22, int ldx22, long theta,
                          long u1, int ldu1, long u2,int ldu2, long v1t, int ldv1t,
                          long v2t, int ldv2t );
        int dorcsint(int matriint, char jobu1, char jobu2,
                            char jobv1t, char jobv2t, char trans,
                            char signs, int m, int p,int q, long x11, int ldx11,
                            long x12, int ldx12, long x21,int ldx21, long x22, int ldx22,
                            long theta, long u1, int ldu1,
                            long u2, int ldu2, long v1t,int ldv1t, long v2t, int ldv2t,
                            long work, int lwork,int iwork );
        int dorcsd2by1(int matriint, char jobu1, char jobu2,
                              char jobv1t, int m, int p, int q,
                              long x11, int ldx11, long x21, int ldx21,
                              long theta, long u1, int ldu1, long u2,int ldu2, long v1t, int ldv1t);
        int dorcsd2by1_work(int matriint, char jobu1, char jobu2,
                                   char jobv1t, int m, int p, int q,
                                   long x11, int ldx11, long x21, int ldx21,
                                   long theta, long u1, int ldu1, long u2,int ldu2, long v1t, int ldv1t,
                                   long work, int lwork, int iwork );
        int dsyconv(int matriint, char uplo, char way, int n,
                           long a, int lda,int ipiv, long work);
        int dsyconint(int matriint, char uplo, char way,int n, long a, int lda,int ipiv, long work );
        int dsyswapr(int matriint, char uplo, int n,
                            long a, int i1, int i2 );
        int dsyswapint(int matriint, char uplo, int n,
                              long a, int i1, int i2 );
        int dsytri2(int matriint, char uplo, int n,
                           long a, int lda,int ipiv );
        int dsytri2_work(int matriint, char uplo, int n,
                                long a, int lda,int ipiv,
                                IComplexNumber work, int lwork );
        int dsytri2x(int matriint, char uplo, int n,
                            long a, int lda,int ipiv,int nb );
        int dsytri2int(int matriint, char uplo, int n,
                              long a, int lda,int ipiv, long work,int nb );
        int dsytrs2(int matriint, char uplo, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
        int dsytrs2_work(int matriint, char uplo, int n,int nrhs,  long a,int lda,int ipiv,
                                long b, int ldb, long work );
        int sbbcsd(int matriint, char jobu1, char jobu2,
                          char jobv1t, char jobv2t, char trans, int m,int p, int q, long theta, long phi,
                          long u1, int ldu1, long u2,int ldu2, long v1t, int ldv1t,
                          long v2t, int ldv2t, long b11d,
                          long b11e, long b12d, long b12e, long b21d,
                          long b21e, long b22d, long b22e );
        int sbbcsint(int matriint, char jobu1, char jobu2,
                            char jobv1t, char jobv2t, char trans,int m, int p, int q,
                            long theta, long phi, long u1,int ldu1, long u2, int ldu2,
                            long v1t, int ldv1t, long v2t,int ldv2t, long b11d, long b11e,
                            long b12d, long b12e, long b21d,
                            long b21e, long b22d, long b22e,
                            long work, int lwork );
        int sorbdb(int matriint, char trans, char signs,int m, int p, int q, long x11,int ldx11, long x12, int ldx12,
                          long x21, int ldx21, long x22,int ldx22, long theta, long phi,
                          long taup1, long taup2, long tauq1,
                          long tauq2 );
        int sorbdint(int matriint, char trans, char signs,int m, int p, int q,
                            long x11, int ldx11, long x12,int ldx12, long x21, int ldx21,
                            long x22, int ldx22, long theta,
                            long phi, long taup1, long taup2,
                            long tauq1, long tauq2, long work,int lwork );
        int sorcsd(int matriint, char jobu1, char jobu2,
                          char jobv1t, char jobv2t, char trans, char signs,int m, int p, int q, long x11,int ldx11, long x12, int ldx12,
                          long x21, int ldx21, long x22,int ldx22, long theta, long u1,int ldu1, long u2, int ldu2,
                          long v1t, int ldv1t, long v2t,int ldv2t );
        int sorcsint(int matriint, char jobu1, char jobu2,
                            char jobv1t, char jobv2t, char trans,
                            char signs, int m, int p,int q, long x11, int ldx11,
                            long x12, int ldx12, long x21,int ldx21, long x22, int ldx22,
                            long theta, long u1, int ldu1,
                            long u2, int ldu2, long v1t,int ldv1t, long v2t, int ldv2t,
                            long work, int lwork,int iwork );
        int sorcsd2by1(int matriint, char jobu1, char jobu2,
                              char jobv1t, int m, int p, int q,
                              long x11, int ldx11, long x21, int ldx21,
                              long theta, long u1, int ldu1, long u2,int ldu2, long v1t, int ldv1t);
        int sorcsd2by1_work(int matriint, char jobu1, char jobu2,
                                   char jobv1t, int m, int p, int q,
                                   long x11, int ldx11, long x21, int ldx21,
                                   long theta, long u1, int ldu1, long u2,int ldu2, long v1t, int ldv1t,
                                   long work, int lwork, int iwork );
        int ssyconv(int matriint, char uplo, char way, int n,
                           long a, int lda,int ipiv, long work );
        int ssyconint(int matriint, char uplo, char way,int n, long a, int lda,int ipiv, long work );
        int ssyswapr(int matriint, char uplo, int n,
                            long a, int i1, int i2 );
        int ssyswapint(int matriint, char uplo, int n,
                              long a, int i1, int i2 );
        int ssytri2(int matriint, char uplo, int n, long a,int lda,int ipiv );
        int ssytri2_work(int matriint, char uplo, int n,
                                long a, int lda,int ipiv,
                                IComplexNumber work, int lwork );
        int ssytri2x(int matriint, char uplo, int n,
                            long a, int lda,int ipiv,int nb );
        int ssytri2int(int matriint, char uplo, int n,
                              long a, int lda,int ipiv, long work,int nb );
        int ssytrs2(int matriint, char uplo, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
        int ssytrs2_work(int matriint, char uplo, int n,int nrhs,  long a,int lda,int ipiv,
                                long b, int ldb, long work );
        int zbbcsd(int matriint, char jobu1, char jobu2,
                          char jobv1t, char jobv2t, char trans, int m,int p, int q, long theta,
                          long phi, IComplexNumber u1,int ldu1, IComplexNumber u2,int ldu2, IComplexNumber v1t,int ldv1t, IComplexNumber v2t,int ldv2t, long b11d, long b11e,
                          long b12d, long b12e, long b21d,
                          long b21e, long b22d, long b22e );
        int zbbcsint(int matriint, char jobu1, char jobu2,
                            char jobv1t, char jobv2t, char trans,int m, int p, int q,
                            long theta, long phi,
                            IComplexNumber u1, int ldu1,
                            IComplexNumber u2, int ldu2,
                            IComplexNumber v1t, int ldv1t,
                            IComplexNumber v2t, int ldv2t,
                            long b11d, long b11e, long b12d,
                            long b12e, long b21d, long b21e,
                            long b22d, long b22e, long rwork,int lrwork );
        int zheswapr(int matriint, char uplo, int n,
                            IComplexNumber a, int i1,int i2 );
        int zheswapint(int matriint, char uplo, int n,
                              IComplexNumber a, int i1,int i2 );
        int zhetri2(int matriint, char uplo, int n,
                           IComplexNumber a, int lda,int ipiv );
        int zhetri2_work(int matriint, char uplo, int n,
                                IComplexNumber a, int lda,int ipiv,
                                IComplexNumber work, int lwork );
        int zhetri2x(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv, int nb );
        int zhetri2int(int matriint, char uplo, int n,
                              IComplexNumber a, int lda,int ipiv,
                              IComplexNumber work, int nb );
        int zhetrs2(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                           IComplexNumber b, int ldb );
        int zhetrs2_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                                IComplexNumber b, int ldb,
                                IComplexNumber work );
        int zsyconv(int matriint, char uplo, char way, int n,
                           IComplexNumber a, int lda,int ipiv, IComplexNumber work );
        int zsyconint(int matriint, char uplo, char way,int n, IComplexNumber a,int lda,int ipiv,
                             IComplexNumber work );
        int zsyswapr(int matriint, char uplo, int n,
                            IComplexNumber a, int i1,int i2 );
        int zsyswapint(int matriint, char uplo, int n,
                              IComplexNumber a, int i1,int i2 );
        int zsytri2(int matriint, char uplo, int n,
                           IComplexNumber a, int lda,int ipiv );
        int zsytri2_work(int matriint, char uplo, int n,
                                IComplexNumber a, int lda,int ipiv,
                                IComplexNumber work, int lwork );
        int zsytri2x(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv, int nb );
        int zsytri2int(int matriint, char uplo, int n,
                              IComplexNumber a, int lda,int ipiv,
                              IComplexNumber work, int nb );
        int zsytrs2(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                           IComplexNumber b, int ldb );
        int zsytrs2_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                                IComplexNumber b, int ldb,
                                IComplexNumber work );
        int zunbdb(int matriint, char trans, char signs,int m, int p, int q,
                          IComplexNumber x11, int ldx11,
                          IComplexNumber x12, int ldx12,
                          IComplexNumber x21, int ldx21,
                          IComplexNumber x22, int ldx22,
                          long theta, long phi,
                          IComplexNumber taup1,
                          IComplexNumber taup2,
                          IComplexNumber tauq1,
                          IComplexNumber tauq2 );
        int zunbdint(int matriint, char trans, char signs,int m, int p, int q,
                            IComplexNumber x11, int ldx11,
                            IComplexNumber x12, int ldx12,
                            IComplexNumber x21, int ldx21,
                            IComplexNumber x22, int ldx22,
                            long theta, long phi,
                            IComplexNumber taup1,
                            IComplexNumber taup2,
                            IComplexNumber tauq1,
                            IComplexNumber tauq2,
                            IComplexNumber work, int lwork );
        int zuncsd(int matriint, char jobu1, char jobu2,
                          char jobv1t, char jobv2t, char trans, char signs,int m, int p, int q,
                          IComplexNumber x11, int ldx11,
                          IComplexNumber x12, int ldx12,
                          IComplexNumber x21, int ldx21,
                          IComplexNumber x22, int ldx22,
                          long theta, IComplexNumber u1,int ldu1, IComplexNumber u2,int ldu2, IComplexNumber v1t,int ldv1t, IComplexNumber v2t,int ldv2t );
        int zuncsint(int matriint, char jobu1, char jobu2,
                            char jobv1t, char jobv2t, char trans,
                            char signs, int m, int p,int q, IComplexNumber x11,int ldx11, IComplexNumber x12,int ldx12, IComplexNumber x21,int ldx21, IComplexNumber x22,int ldx22, long theta,
                            IComplexNumber u1, int ldu1,
                            IComplexNumber u2, int ldu2,
                            IComplexNumber v1t, int ldv1t,
                            IComplexNumber v2t, int ldv2t,
                            IComplexNumber work, int lwork,
                            long rwork, int lrwork,int iwork );
        int zuncsd2by1(int matriint, char jobu1, char jobu2,
                              char jobv1t, int m, int p, int q,
                              IComplexNumber x11, int ldx11,
                              IComplexNumber x21, int ldx21,
                              IComplexNumber theta, IComplexNumber u1,int ldu1, IComplexNumber u2,int ldu2, IComplexNumber v1t, int ldv1t );
        int zuncsd2by1_work(int matriint, char jobu1, char jobu2,
                                   char jobv1t, int m, int p,int q, IComplexNumber x11, int ldx11,
                                   IComplexNumber x21, int ldx21,
                                   IComplexNumber theta, IComplexNumber u1,int ldu1, IComplexNumber u2,int ldu2, IComplexNumber v1t,int ldv1t, IComplexNumber work,int lwork, long rwork, int lrwork,int iwork );

//LAPACK 3.4.0
        int sgemqrt(int matriint, char side, char trans,int m, int n, int k,int nb,  long v, int ldv,
                           long t, int ldt, long c,int ldc );
        int dgemqrt(int matriint, char side, char trans,int m, int n, int k,int nb,  long v, int ldv,
                           long t, int ldt, long c,int ldc );
        int cgemqrt(int matriint, char side, char trans,int m, int n, int k,int nb,  IComplexNumber v,int ldv,  IComplexNumber t,int ldt, IComplexNumber c,int ldc );
        int zgemqrt(int matriint, char side, char trans,int m, int n, int k,int nb,  IComplexNumber v,int ldv,  IComplexNumber t,int ldt, IComplexNumber c,int ldc );

        int sgeqrt(int matriint, int m, int n,int nb, long a, int lda, long t,int ldt );
        int dgeqrt(int matriint, int m, int n,int nb, long a, int lda, long t,int ldt );
        int cgeqrt(int matriint, int m, int n,int nb, IComplexNumber a,int lda, IComplexNumber t,int ldt );
        int zgeqrt(int matriint, int m, int n,int nb, IComplexNumber a,int lda, IComplexNumber t,int ldt );

        int sgeqrt2(int matriint, int m, int n,
                           long a, int lda, long t,int ldt );
        int dgeqrt2(int matriint, int m, int n,
                           long a, int lda, long t,int ldt );
        int cgeqrt2(int matriint, int m, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber t, int ldt );
        int zgeqrt2(int matriint, int m, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber t, int ldt );

        int sgeqrt3(int matriint, int m, int n,
                           long a, int lda, long t,int ldt );
        int dgeqrt3(int matriint, int m, int n,
                           long a, int lda, long t,int ldt );
        int cgeqrt3(int matriint, int m, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber t, int ldt );
        int zgeqrt3(int matriint, int m, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber t, int ldt );

        int stpmqrt(int matriint, char side, char trans,int m, int n, int k,int l, int nb,  long v,int ldv,  long t, int ldt,
                           long a, int lda, long b,int ldb );
        int dtpmqrt(int matriint, char side, char trans,int m, int n, int k,int l, int nb,  long v,int ldv,  long t, int ldt,
                           long a, int lda, long b,int ldb );
        int ctpmqrt(int matriint, char side, char trans,int m, int n, int k,int l, int nb,
                           IComplexNumber v, int ldv,
                           IComplexNumber t, int ldt,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb );
        int ztpmqrt(int matriint, char side, char trans,int m, int n, int k,int l, int nb,
                           IComplexNumber v, int ldv,
                           IComplexNumber t, int ldt,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb );

        int stpqrt(int matriint, int m, int n,int l, int nb, long a,int lda, long b, int ldb, long t,int ldt );

        int dtpqrt(int matriint, int m, int n,int l, int nb, long a,int lda, long b, int ldb, long t,int ldt );
        int ctpqrt(int matriint, int m, int n,int l, int nb,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber t, int ldt );
        int ztpqrt(int matriint, int m, int n,int l, int nb,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber t, int ldt );

        int stpqrt2(int matriint,int m, int n, int l,
                           long a, int lda,
                           long b, int ldb,
                           long t, int ldt );
        int dtpqrt2(int matriint,int m, int n, int l,
                           long a, int lda,
                           long b, int ldb,
                           long t, int ldt );
        int ctpqrt2(int matriint,int m, int n, int l,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,
                           IComplexNumber t, int ldt );
        int ztpqrt2(int matriint,int m, int n, int l,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,
                           IComplexNumber t, int ldt );

        int stprfb(int matriint, char side, char trans, char direct,
                          char storev, int m, int n,int k, int l,  long v,int ldv,  long t, int ldt,
                          long a, int lda, long b, int ldb );
        int dtprfb(int matriint, char side, char trans, char direct,
                          char storev, int m, int n,int k, int l,  long v,int ldv,  long t, int ldt,
                          long a, int lda, long b, int ldb );
        int ctprfb(int matriint, char side, char trans, char direct,
                          char storev, int m, int n,int k, int l,
                          IComplexNumber v, int ldv,
                          IComplexNumber t, int ldt,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb );
        int ztprfb(int matriint, char side, char trans, char direct,
                          char storev, int m, int n,int k, int l,
                          IComplexNumber v, int ldv,
                          IComplexNumber t, int ldt,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb );

        int sgemqrint(int matriint, char side, char trans,int m, int n, int k,int nb,  long v, int ldv,
                             long t, int ldt, long c,int ldc, long work );
        int dgemqrint(int matriint, char side, char trans,int m, int n, int k,int nb,  long v, int ldv,
                             long t, int ldt, long c,int ldc, long work );
        int cgemqrint(int matriint, char side, char trans,int m, int n, int k,int nb,  IComplexNumber v,int ldv,  IComplexNumber t,int ldt, IComplexNumber c,int ldc, IComplexNumber work );
        int zgemqrint(int matriint, char side, char trans,int m, int n, int k,int nb,  IComplexNumber v,int ldv,  IComplexNumber t,int ldt, IComplexNumber c,int ldc, IComplexNumber work );

        int sgeqrint(int matriint, int m, int n,int nb, long a, int lda,
                            long t, int ldt, long work );
        int dgeqrint(int matriint, int m, int n,int nb, long a, int lda,
                            long t, int ldt, long work );
        int cgeqrint(int matriint, int m, int n,int nb, IComplexNumber a,int lda, IComplexNumber t,int ldt, IComplexNumber work );
        int zgeqrint(int matriint, int m, int n,int nb, IComplexNumber a,int lda, IComplexNumber t,int ldt, IComplexNumber work );

        int sgeqrt2_work(int matriint, int m, int n,
                                long a, int lda, long t,int ldt );
        int dgeqrt2_work(int matriint, int m, int n,
                                long a, int lda, long t,int ldt );
        int cgeqrt2_work(int matriint, int m, int n,
                                IComplexNumber a, int lda,
                                IComplexNumber t, int ldt );
        int zgeqrt2_work(int matriint, int m, int n,
                                IComplexNumber a, int lda,
                                IComplexNumber t, int ldt );

        int sgeqrt3_work(int matriint, int m, int n,
                                long a, int lda, long t,int ldt );
        int dgeqrt3_work(int matriint, int m, int n,
                                long a, int lda, long t,int ldt );
        int cgeqrt3_work(int matriint, int m, int n,
                                IComplexNumber a, int lda,
                                IComplexNumber t, int ldt );
        int zgeqrt3_work(int matriint, int m, int n,
                                IComplexNumber a, int lda,
                                IComplexNumber t, int ldt );

        int stpmqrint(int matriint, char side, char trans,int m, int n, int k,int l, int nb,  long v,int ldv,  long t, int ldt,
                             long a, int lda, long b,int ldb, long work );
        int dtpmqrint(int matriint, char side, char trans,int m, int n, int k,int l, int nb,  long v,int ldv,  long t,int ldt, long a, int lda,
                             long b, int ldb, long work );
        int ctpmqrint(int matriint, char side, char trans,int m, int n, int k,int l, int nb,
                             IComplexNumber v, int ldv,
                             IComplexNumber t, int ldt,
                             IComplexNumber a, int lda,
                             IComplexNumber b, int ldb,
                             IComplexNumber work );
        int ztpmqrint(int matriint, char side, char trans,int m, int n, int k,int l, int nb,
                             IComplexNumber v, int ldv,
                             IComplexNumber t, int ldt,
                             IComplexNumber a, int lda,
                             IComplexNumber b, int ldb,
                             IComplexNumber work );

        int stpqrint(int matriint, int m, int n,int l, int nb, long a,int lda, long b, int ldb,
                            long t, int ldt, long work );
        int dtpqrint(int matriint, int m, int n,int l, int nb, long a,int lda, long b, int ldb,
                            long t, int ldt, long work );
        int ctpqrint(int matriint, int m, int n,int l, int nb,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber t, int ldt,
                            IComplexNumber work );
        int ztpqrint(int matriint, int m, int n,int l, int nb,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber t, int ldt,
                            IComplexNumber work );

        int stpqrt2_work(int matriint,int m, int n, int l,
                                long a, int lda,
                                long b, int ldb,
                                long t, int ldt );
        int dtpqrt2_work(int matriint,int m, int n, int l,
                                long a, int lda,
                                long b, int ldb,
                                long t, int ldt );
        int ctpqrt2_work(int matriint,int m, int n, int l,
                                IComplexNumber a, int lda,
                                IComplexNumber b, int ldb,
                                IComplexNumber t, int ldt );
        int ztpqrt2_work(int matriint,int m, int n, int l,
                                IComplexNumber a, int lda,
                                IComplexNumber b, int ldb,
                                IComplexNumber t, int ldt );

        int stprfint(int matriint, char side, char trans,
                            char direct, char storev, int m,int n, int k, int l,
                            long v, int ldv,  long t,int ldt, long a, int lda,
                            long b, int ldb,  long work,int ldwork );
        int dtprfint(int matriint, char side, char trans,
                            char direct, char storev, int m,int n, int k, int l,
                            long v, int ldv,
                            long t, int ldt, long a,int lda, long b, int ldb,
                            long work, int ldwork );
        int ctprfint(int matriint, char side, char trans,
                            char direct, char storev, int m,int n, int k, int l,
                            IComplexNumber v, int ldv,
                            IComplexNumber t, int ldt,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber work, int ldwork );
        int ztprfint(int matriint, char side, char trans,
                            char direct, char storev, int m,int n, int k, int l,
                            IComplexNumber v, int ldv,
                            IComplexNumber t, int ldt,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber work, int ldwork );
//LAPACK 3.X.X
        int ssysint(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb );
        int dsysint(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb );
        int csysint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                           IComplexNumber b, int ldb );
        int zsysint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                           IComplexNumber b, int ldb );

        int ssytrint(int matriint, char uplo, int n, long a,int lda, int ipiv );
        int dsytrint(int matriint, char uplo, int n, long a,int lda, int ipiv );
        int csytrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv );
        int zsytrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv );

        int ssytrint(int matriint, char uplo, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
        int dsytrint(int matriint, char uplo, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
        int csytrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                            IComplexNumber b, int ldb );
        int zsytrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                            IComplexNumber b, int ldb );

        int chetrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv );
        int zhetrint(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv );

        int chetrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                            IComplexNumber b, int ldb );
        int zhetrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                            IComplexNumber b, int ldb );

        int csyr(int matriint, char uplo, int n,
                        IComplexNumber alpha,
                        IComplexNumber x, int incx,
                        IComplexNumber a, int lda );
        int zsyr(int matriint, char uplo, int n,
                        IComplexNumber alpha,
                        IComplexNumber x, int incx,
                        IComplexNumber a, int lda );

        int ssysint_work(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb,
                                long work, int lwork );
        int dsysint_work(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb,
                                long work, int lwork );
        int csysint_work(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                                IComplexNumber b, int ldb,
                                IComplexNumber work,int lwork );
        int zsysint_work(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                                IComplexNumber b, int ldb,
                                IComplexNumber work,int lwork );

        int ssytrint_work(int matriint, char uplo, int n,
                                 long a, int lda, int ipiv,
                                 long work, int lwork );
        int dsytrint_work(int matriint, char uplo, int n,
                                 long a, int lda, int ipiv,
                                 long work, int lwork );
        int csytrint_work(int matriint, char uplo, int n,
                                 IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );
        int zsytrint_work(int matriint, char uplo, int n,
                                 IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );

        int ssytrint_work(int matriint, char uplo, int n,int nrhs,  long a, int lda,int ipiv, long b,int ldb );
        int dsytrint_work(int matriint, char uplo, int n,int nrhs,  long a,int lda,int ipiv,
                                 long b, int ldb );
        int csytrint_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                                 IComplexNumber b, int ldb );
        int zsytrint_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                                 IComplexNumber b, int ldb );

        int chetrint_work(int matriint, char uplo, int n,
                                 IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );
        int zhetrint_work(int matriint, char uplo, int n,
                                 IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );

        int chetrint_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                                 IComplexNumber b, int ldb );
        int zhetrint_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                                 IComplexNumber b, int ldb );


        int csyint(int matriint, char uplo, int n,
                          IComplexNumber alpha,
                          IComplexNumber x,int incx, IComplexNumber a,int lda );
        int zsyint(int matriint, char uplo, int n,
                          IComplexNumber alpha,
                          IComplexNumber x,int incx, IComplexNumber a,int lda );

        void dgetrf(int m, int n, long a, int lda,int ipiv, int info );
        void cgetrf(int m, int n, IComplexNumber a,int lda, int ipiv, int info );
        void zgetrf(int m, int n, IComplexNumber a,int lda, int ipiv, int info );
        void sgetrf2(int m, int n, long a, int lda,int ipiv, int info );
        void dgetrf2(int m, int n, long a, int lda,int ipiv, int info );
        void cgetrf2(int m, int n, IComplexNumber a,int lda, int ipiv, int info );
        void zgetrf2(int m, int n, IComplexNumber a,int lda, int ipiv, int info );
        void sgbtrf(int m, int n, int kl,int ku, long ab, int ldab,int ipiv, int info );
        void dgbtrf(int m, int n, int kl,int ku, long ab, int ldab,int ipiv, int info );
        void cgbtrf(int m, int n, int kl,int ku, IComplexNumber ab, int ldab,int ipiv, int info );
        void zgbtrf(int m, int n, int kl,int ku, IComplexNumber ab, int ldab,int ipiv, int info );
        void sgttrf(int n, long dl, long d, long du, long du2,int ipiv, int info );
        void dgttrf(int n, long dl, long d, long du,
                           long du2, int ipiv, int info );
        void cgttrf(int n, IComplexNumber dl,
                           IComplexNumber d, IComplexNumber du,
                           IComplexNumber du2, int ipiv,int info );
        void zgttrf(int n, IComplexNumber dl,
                           IComplexNumber d, IComplexNumber du,
                           IComplexNumber du2, int ipiv,int info );
        void spotrf2( String uplo, int n, long a, int lda,int info );
        void dpotrf2( String uplo, int n, long a, int lda,int info );
        void cpotrf2( String uplo, int n, IComplexNumber a,int lda, int info );
        void zpotrf2( String uplo, int n, IComplexNumber a,int lda, int info );
        void spotrf( String uplo, int n, long a, int lda,int info );
        void dpotrf( String uplo, int n, long a, int lda,int info );
        void cpotrf( String uplo, int n, IComplexNumber a,int lda, int info );
        void zpotrf( String uplo, int n, IComplexNumber a,int lda, int info );
        void dpstrf( String uplo, int n, long a, int lda,int piv, int rank, long tol,
                            long work, int info );
        void spstrf( String uplo, int n, long a, int lda,int piv, int rank, long tol, long work,int info );
        void zpstrf( String uplo, int n, IComplexNumber a,int lda, int piv, int rank,
                            long tol, long work, int info );
        void cpstrf( String uplo, int n, IComplexNumber a,int lda, int piv, int rank,
                            long tol, long work, int info );
        void dpftrf( String transr, String uplo, int n, long a,int info );
        void spftrf( String transr, String uplo, int n, long a,int info );
        void zpftrf( String transr, String uplo, int n,
                            IComplexNumber a, int info );
        void cpftrf( String transr, String uplo, int n,
                            IComplexNumber a, int info );
        void spptrf( String uplo, int n, long ap, int info );
        void dpptrf( String uplo, int n, long ap, int info );
        void cpptrf( String uplo, int n, IComplexNumber ap,int info );
        void zpptrf( String uplo, int n, IComplexNumber ap,int info );
        void spbtrf( String uplo, int n, int kd, long ab,int ldab, int info );
        void dpbtrf( String uplo, int n, int kd, long ab,int ldab, int info );
        void cpbtrf( String uplo, int n, int kd,
                            IComplexNumber ab, int ldab,int info );
        void zpbtrf( String uplo, int n, int kd,
                            IComplexNumber ab, int ldab,int info );
        void spttrf(int n, long d, long e, int info );
        void dpttrf(int n, long d, long e, int info );
        void cpttrf(int n, long d, IComplexNumber e,int info );
        void zpttrf(int n, long d, IComplexNumber e,int info );
        void ssytrf( String uplo, int n, long a, int lda,int ipiv, long work, int lwork,int info );
        void dsytrf( String uplo, int n, long a, int lda,int ipiv, long work, int lwork,int info );
        void csytrf( String uplo, int n, IComplexNumber a,int lda, int ipiv,
                            IComplexNumber work, int lwork,int info );
        void zsytrf( String uplo, int n, IComplexNumber a,int lda, int ipiv,
                            IComplexNumber work, int lwork,int info );
        void chetrf( String uplo, int n, IComplexNumber a,int lda, int ipiv,
                            IComplexNumber work, int lwork,int info );
        void zhetrf( String uplo, int n, IComplexNumber a,int lda, int ipiv,
                            IComplexNumber work, int lwork,int info );
        void ssptrf( String uplo, int n, long ap, int ipiv,int info );
        void dsptrf( String uplo, int n, long ap, int ipiv,int info );
        void csptrf( String uplo, int n, IComplexNumber ap,int ipiv, int info );
        void zsptrf( String uplo, int n, IComplexNumber ap,int ipiv, int info );
        void chptrf( String uplo, int n, IComplexNumber ap,int ipiv, int info );
        void zhptrf( String uplo, int n, IComplexNumber ap,int ipiv, int info );
        void sgetrs( String trans, int n, int nrhs,
                            long a, int lda,int ipiv,
                            long b, int ldb, int info );
        void dgetrs( String trans, int n, int nrhs,
                            long a, int lda,int ipiv,
                            long b, int ldb, int info );
        void cgetrs( String trans, int n, int nrhs,
                            IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, int info );
        void zgetrs( String trans, int n, int nrhs,
                            IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, int info );
        void sgbtrs( String trans, int n, int kl, int ku,int nrhs,  long ab, int ldab,int ipiv, long b, int ldb,int info );
        void dgbtrs( String trans, int n, int kl, int ku,int nrhs,  long ab, int ldab,int ipiv, long b, int ldb,int info );
        void cgbtrs( String trans, int n, int kl, int ku,int nrhs,  IComplexNumber ab,int ldab,int ipiv,
                            IComplexNumber b, int ldb,int info );
        void zgbtrs( String trans, int n, int kl, int ku,int nrhs,  IComplexNumber ab,int ldab,int ipiv,
                            IComplexNumber b, int ldb,int info );
        void sgttrs( String trans, int n, int nrhs,
                            long dl,  long d,  long du,
                            long du2,int ipiv, long b,int ldb, int info );
        void dgttrs( String trans, int n, int nrhs,
                            long dl,  long d,  long du,
                            long du2,int ipiv, long b,int ldb, int info );
        void cgttrs( String trans, int n, int nrhs,
                            IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du,
                            IComplexNumber du2,int ipiv,
                            IComplexNumber b, int ldb,int info );
        void zgttrs( String trans, int n, int nrhs,
                            IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du,
                            IComplexNumber du2,int ipiv,
                            IComplexNumber b, int ldb,int info );
        void spotrs( String uplo, int n, int nrhs,  long a,int lda, long b, int ldb,int info );
        void dpotrs( String uplo, int n, int nrhs,
                            long a, int lda, long b,int ldb, int info );
        void cpotrs( String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,int info );
        void zpotrs( String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,int info );
        void dpftrs( String transr, String uplo, int n, int nrhs,
                            long a, long b, int ldb,int info );
        void spftrs( String transr, String uplo, int n, int nrhs,
                            long a, long b, int ldb,int info );
        void zpftrs( String transr, String uplo, int n, int nrhs,
                            IComplexNumber a, IComplexNumber b,int ldb, int info );
        void cpftrs( String transr, String uplo, int n, int nrhs,
                            IComplexNumber a, IComplexNumber b,int ldb, int info );
        void spptrs( String uplo, int n, int nrhs,
                            long ap, long b, int ldb,int info );
        void dpptrs( String uplo, int n, int nrhs,
                            long ap, long b, int ldb,int info );
        void cpptrs( String uplo, int n, int nrhs,
                            IComplexNumber ap, IComplexNumber b,int ldb, int info );
        void zpptrs( String uplo, int n, int nrhs,
                            IComplexNumber ap, IComplexNumber b,int ldb, int info );
        void spbtrs( String uplo, int n, int kd, int nrhs,
                            long ab, int ldab, long b,int ldb, int info );
        void dpbtrs( String uplo, int n, int kd, int nrhs,
                            long ab, int ldab, long b,int ldb, int info );
        void cpbtrs( String uplo, int n, int kd, int nrhs,
                            IComplexNumber ab, int ldab,
                            IComplexNumber b, int ldb,int info );
        void zpbtrs( String uplo, int n, int kd, int nrhs,
                            IComplexNumber ab, int ldab,
                            IComplexNumber b, int ldb,int info );
        void spttrs(int n, int nrhs,  long d,
                           long e, long b, int ldb,int info );
        void dpttrs(int n, int nrhs,  long d,
                           long e, long b, int ldb,int info );
        void cpttrs( String uplo, int n, int nrhs,  long d,
                            IComplexNumber e, IComplexNumber b,int ldb, int info );
        void zpttrs( String uplo, int n, int nrhs,
                            long d,  IComplexNumber e,
                            IComplexNumber b, int ldb,int info );
        void ssytrs( String uplo, int n, int nrhs,  long a,int lda,int ipiv, long b,int ldb, int info );
        void dsytrs( String uplo, int n, int nrhs,
                            long a, int lda,int ipiv,
                            long b, int ldb, int info );
        void csytrs( String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, int info );
        void zsytrs( String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, int info );
        void chetrs( String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, int info );
        void zhetrs( String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, int info );
        void ssptrs( String uplo, int n, int nrhs,
                            long ap,int ipiv, long b,int ldb, int info );
        void dsptrs( String uplo, int n, int nrhs,
                            long ap,int ipiv, long b,int ldb, int info );
        void csptrs( String uplo, int n, int nrhs,
                            IComplexNumber ap,int ipiv,
                            IComplexNumber b, int ldb,int info );
        void zsptrs( String uplo, int n, int nrhs,
                            IComplexNumber ap,int ipiv,
                            IComplexNumber b, int ldb,int info );
        void chptrs( String uplo, int n, int nrhs,
                            IComplexNumber ap,int ipiv,
                            IComplexNumber b, int ldb,int info );
        void zhptrs( String uplo, int n, int nrhs,
                            IComplexNumber ap,int ipiv,
                            IComplexNumber b, int ldb,int info );
        void strtrs( String uplo, String trans, String diag, int n,int nrhs,  long a, int lda, long b,int ldb, int info );
        void dtrtrs( String uplo, String trans, String diag, int n,int nrhs,  long a, int lda,
                            long b, int ldb, int info );
        void ctrtrs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber b, int ldb,int info );
        void ztrtrs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber b, int ldb,int info );
        void stptrs( String uplo, String trans, String diag, int n,int nrhs,  long ap, long b,int ldb, int info );
        void dtptrs( String uplo, String trans, String diag, int n,int nrhs,  long ap, long b,int ldb, int info );
        void ctptrs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber ap,
                            IComplexNumber b, int ldb,int info );
        void ztptrs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber ap,
                            IComplexNumber b, int ldb,int info );
        void stbtrs( String uplo, String trans, String diag, int n,int kd, int nrhs,  long ab,int ldab, long b, int ldb,int info );
        void dtbtrs( String uplo, String trans, String diag, int n,int kd, int nrhs,  long ab,int ldab, long b, int ldb,int info );
        void ctbtrs( String uplo, String trans, String diag, int n,int kd, int nrhs,
                            IComplexNumber ab, int ldab,
                            IComplexNumber b, int ldb,int info );
        void ztbtrs( String uplo, String trans, String diag, int n,int kd, int nrhs,
                            IComplexNumber ab, int ldab,
                            IComplexNumber b, int ldb,int info );
        void sgecon( String norm, int n,  long a, int lda,
                            long anorm, long rcond, long work, int iwork,int info );
        void dgecon( String norm, int n,  long a, int lda,
                            long anorm, long rcond, long work,int iwork, int info );
        void cgecon( String norm, int n,  IComplexNumber a,int lda, long anorm, long rcond,
                            IComplexNumber work, long rwork,int info );
        void zgecon( String norm, int n,  IComplexNumber a,int lda, long anorm, long rcond,
                            IComplexNumber work, long rwork,int info );
        void sgbcon( String norm, int n, int kl, int ku,
                            long ab, int ldab,int ipiv,
                            long anorm, long rcond, long work, int iwork,int info );
        void dgbcon( String norm, int n, int kl, int ku,
                            long ab, int ldab,int ipiv,
                            long anorm, long rcond, long work,int iwork, int info );
        void cgbcon( String norm, int n, int kl, int ku,
                            IComplexNumber ab, int ldab,int ipiv, long anorm, long rcond,
                            IComplexNumber work, long rwork,int info );
        void zgbcon( String norm, int n, int kl, int ku,
                            IComplexNumber ab, int ldab,int ipiv, long anorm, long rcond,
                            IComplexNumber work, long rwork,int info );
        void sgtcon( String norm, int n,  long dl,  long d,
                            long du,  long du2,int ipiv,
                            long anorm, long rcond, long work, int iwork,int info );
        void dgtcon( String norm, int n,  long dl,
                            long d,  long du,  long du2,int ipiv, long anorm, long rcond,
                            long work, int iwork, int info );
        void cgtcon( String norm, int n,  IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du,
                            IComplexNumber du2,int ipiv,
                            long anorm, long rcond, IComplexNumber work,int info );
        void zgtcon( String norm, int n,  IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du,
                            IComplexNumber du2,int ipiv,
                            long anorm, long rcond, IComplexNumber work,int info );
        void spocon( String uplo, int n,  long a, int lda,
                            long anorm, long rcond, long work, int iwork,int info );
        void dpocon( String uplo, int n,  long a, int lda,
                            long anorm, long rcond, long work,int iwork, int info );
        void cpocon( String uplo, int n,  IComplexNumber a,int lda, long anorm, long rcond,
                            IComplexNumber work, long rwork,int info );
        void zpocon( String uplo, int n,  IComplexNumber a,int lda, long anorm, long rcond,
                            IComplexNumber work, long rwork,int info );
        void sppcon( String uplo, int n,  long ap, long anorm,
                            long rcond, long work, int iwork,int info );
        void dppcon( String uplo, int n,  long ap, long anorm,
                            long rcond, long work, int iwork,int info );
        void cppcon( String uplo, int n,  IComplexNumber ap,
                            long anorm, long rcond, IComplexNumber work,
                            long rwork, int info );
        void zppcon( String uplo, int n,  IComplexNumber ap,
                            long anorm, long rcond, IComplexNumber work,
                            long rwork, int info );
        void spbcon( String uplo, int n, int kd,  long ab,int ldab, long anorm, long rcond, long work,int iwork, int info );
        void dpbcon( String uplo, int n, int kd,  long ab,int ldab, long anorm, long rcond,
                            long work, int iwork, int info );
        void cpbcon( String uplo, int n, int kd,
                            IComplexNumber ab, int ldab,
                            long anorm, long rcond, IComplexNumber work,
                            long rwork, int info );
        void zpbcon( String uplo, int n, int kd,
                            IComplexNumber ab, int ldab,
                            long anorm, long rcond, IComplexNumber work,
                            long rwork, int info );
        void sptcon(int n,  long d,  long e, long anorm,
                           long rcond, long work, int info );
        void dptcon(int n,  long d,  long e,
                           long anorm, long rcond, long work,int info );
        void cptcon(int n,  long d,
                           IComplexNumber e, long anorm, long rcond,
                           long work, int info );
        void zptcon(int n,  long d,
                           IComplexNumber e, long anorm,
                           long rcond, long work, int info );
        void ssycon( String uplo, int n,  long a, int lda,int ipiv, long anorm, long rcond,
                            long work, int iwork, int info );
        void dsycon( String uplo, int n,  long a, int lda,int ipiv, long anorm, long rcond,
                            long work, int iwork, int info );
        void csycon( String uplo, int n,  IComplexNumber a,int lda,int ipiv, long anorm,
                            long rcond, IComplexNumber work,int info );
        void zsycon( String uplo, int n,  IComplexNumber a,int lda,int ipiv, long anorm,
                            long rcond, IComplexNumber work,int info );
        void checon( String uplo, int n,  IComplexNumber a,int lda,int ipiv, long anorm,
                            long rcond, IComplexNumber work,int info );
        void zhecon( String uplo, int n,  IComplexNumber a,int lda,int ipiv, long anorm,
                            long rcond, IComplexNumber work,int info );
        void sspcon( String uplo, int n,  long ap,int ipiv, long anorm, long rcond,
                            long work, int iwork, int info );
        void dspcon( String uplo, int n,  long ap,int ipiv, long anorm, long rcond,
                            long work, int iwork, int info );
        void cspcon( String uplo, int n,  IComplexNumber ap,int ipiv, long anorm, long rcond,
                            IComplexNumber work, int info );
        void zspcon( String uplo, int n,  IComplexNumber ap,int ipiv, long anorm, long rcond,
                            IComplexNumber work, int info );
        void chpcon( String uplo, int n,  IComplexNumber ap,int ipiv, long anorm, long rcond,
                            IComplexNumber work, int info );
        void zhpcon( String uplo, int n,  IComplexNumber ap,int ipiv, long anorm, long rcond,
                            IComplexNumber work, int info );
        void strcon( String norm, String uplo, String diag, int n,
                            long a, int lda, long rcond, long work,int iwork, int info );
        void dtrcon( String norm, String uplo, String diag, int n,
                            long a, int lda, long rcond,
                            long work, int iwork, int info );
        void ctrcon( String norm, String uplo, String diag, int n,
                            IComplexNumber a, int lda,
                            long rcond, IComplexNumber work, long rwork,int info );
        void ztrcon( String norm, String uplo, String diag, int n,
                            IComplexNumber a, int lda,
                            long rcond, IComplexNumber work, long rwork,int info );
        void stpcon( String norm, String uplo, String diag, int n,
                            long ap, long rcond, long work,int iwork, int info );
        void dtpcon( String norm, String uplo, String diag, int n,
                            long ap, long rcond, long work,int iwork, int info );
        void ctpcon( String norm, String uplo, String diag, int n,
                            IComplexNumber ap, long rcond,
                            IComplexNumber work, long rwork,int info );
        void ztpcon( String norm, String uplo, String diag, int n,
                            IComplexNumber ap, long rcond,
                            IComplexNumber work, long rwork,int info );
        void stbcon( String norm, String uplo, String diag, int n,int kd,  long ab, int ldab,
                            long rcond, long work, int iwork,int info );
        void dtbcon( String norm, String uplo, String diag, int n,int kd,  long ab, int ldab,
                            long rcond, long work, int iwork,int info );
        void ctbcon( String norm, String uplo, String diag, int n,int kd,  IComplexNumber ab,int ldab, long rcond, IComplexNumber work,
                            long rwork, int info );
        void ztbcon( String norm, String uplo, String diag, int n,int kd,  IComplexNumber ab,int ldab, long rcond,
                            IComplexNumber work, long rwork,int info );
        void sgerfs( String trans, int n, int nrhs,
                            long a, int lda,  long af,int ldaf,int ipiv,  long b,int ldb, long x, int ldx, long ferr,
                            long berr, long work, int iwork,int info );
        void dgerfs( String trans, int n, int nrhs,
                            long a, int lda,  long af,int ldaf,int ipiv,  long b,int ldb, long x, int ldx, long ferr,
                            long berr, long work, int iwork,int info );
        void cgerfs( String trans, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void zgerfs( String trans, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void dgerfsx( String trans, String equed, int n, int nrhs,
                             long a, int lda,  long af,int ldaf,int ipiv,  long r,
                             long c,  long b, int ldb,
                             long x, int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams, long params,
                             long work, int iwork, int info );
        void sgerfsx( String trans, String equed, int n, int nrhs,
                             long a, int lda,  long af,int ldaf,int ipiv,  long r,
                             long c,  long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams, long params,
                             long work, int iwork, int info );
        void zgerfsx( String trans, String equed, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv,  long r,  long c,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void cgerfsx( String trans, String equed, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv,  long r,  long c,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long berr, int int_bnds, long erint_norm,
                             long erint_comp, int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void sgbrfs( String trans, int n, int kl, int ku,int nrhs,  long ab, int ldab,
                            long afb, int ldafb,int ipiv,
                            long b, int ldb, long x, int ldx,
                            long ferr, long berr, long work, int iwork,int info );
        void dgbrfs( String trans, int n, int kl, int ku,int nrhs,  long ab, int ldab,
                            long afb, int ldafb,int ipiv,  long b, int ldb,
                            long x, int ldx, long ferr, long berr,
                            long work, int iwork, int info );
        void cgbrfs( String trans, int n, int kl, int ku,int nrhs,  IComplexNumber ab,int ldab,  IComplexNumber afb,int ldafb,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void zgbrfs( String trans, int n, int kl, int ku,int nrhs,  IComplexNumber ab,int ldab,  IComplexNumber afb,int ldafb,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void dgbrfsx( String trans, String equed, int n, int kl,int ku, int nrhs,  long ab,int ldab,  long afb, int ldafb,int ipiv,  long r,  long c,
                             long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams, long params,
                             long work, int iwork, int info );
        void sgbrfsx( String trans, String equed, int n, int kl,int ku, int nrhs,  long ab,int ldab,  long afb, int ldafb,int ipiv,  long r,  long c,
                             long b, int ldb, long x, int ldx,
                             long rcond, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params, long work,int iwork, int info );
        void zgbrfsx( String trans, String equed, int n, int kl,int ku, int nrhs,
                             IComplexNumber ab, int ldab,
                             IComplexNumber afb, int ldafb,int ipiv,  long r,  long c,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void cgbrfsx( String trans, String equed, int n, int kl,int ku, int nrhs,
                             IComplexNumber ab, int ldab,
                             IComplexNumber afb, int ldafb,int ipiv,  long r,  long c,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long berr, int int_bnds, long erint_norm,
                             long erint_comp, int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void sgtrfs( String trans, int n, int nrhs,
                            long dl,  long d,  long du,
                            long dlf,  long df,  long duf,
                            long du2,int ipiv,  long b,int ldb, long x, int ldx, long ferr,
                            long berr, long work, int iwork,int info );
        void dgtrfs( String trans, int n, int nrhs,
                            long dl,  long d,  long du,
                            long dlf,  long df,  long duf,
                            long du2,int ipiv,  long b,int ldb, long x, int ldx, long ferr,
                            long berr, long work, int iwork,int info );
        void cgtrfs( String trans, int n, int nrhs,
                            IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du,
                            IComplexNumber dlf,
                            IComplexNumber df,
                            IComplexNumber duf,
                            IComplexNumber du2,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void zgtrfs( String trans, int n, int nrhs,
                            IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du,
                            IComplexNumber dlf,
                            IComplexNumber df,
                            IComplexNumber duf,
                            IComplexNumber du2,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void sporfs( String uplo, int n, int nrhs,  long a,int lda,  long af, int ldaf,
                            long b, int ldb, long x, int ldx,
                            long ferr, long berr, long work, int iwork,int info );
        void dporfs( String uplo, int n, int nrhs,
                            long a, int lda,  long af,int ldaf,  long b, int ldb,
                            long x, int ldx, long ferr, long berr,
                            long work, int iwork, int info );
        void cporfs( String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void zporfs( String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void dporfsx( String uplo, String equed, int n, int nrhs,
                             long a, int lda,  long af,int ldaf,  long s,  long b,int ldb, long x, int ldx, long rcond,
                             long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params, long work,int iwork, int info );
        void sporfsx( String uplo, String equed, int n, int nrhs,
                             long a, int lda,  long af,int ldaf,  long s,  long b,int ldb, long x, int ldx, long rcond,
                             long berr, int int_bnds, long erint_norm,
                             long erint_comp, int nparams, long params,
                             long work, int iwork, int info );
        void zporfsx( String uplo, String equed, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,
                             long s,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                             long rcond, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void cporfsx( String uplo, String equed, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,
                             long s,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                             long rcond, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void spprfs( String uplo, int n, int nrhs,
                            long ap,  long afp,  long b,int ldb, long x, int ldx, long ferr,
                            long berr, long work, int iwork,int info );
        void dpprfs( String uplo, int n, int nrhs,
                            long ap,  long afp,  long b,int ldb, long x, int ldx, long ferr,
                            long berr, long work, int iwork,int info );
        void cpprfs( String uplo, int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber afp,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void zpprfs( String uplo, int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber afp,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void spbrfs( String uplo, int n, int kd, int nrhs,
                            long ab, int ldab,  long afb,int ldafb,  long b, int ldb,
                            long x, int ldx, long ferr, long berr,
                            long work, int iwork, int info );
        void dpbrfs( String uplo, int n, int kd, int nrhs,
                            long ab, int ldab,  long afb,int ldafb,  long b, int ldb,
                            long x, int ldx, long ferr, long berr,
                            long work, int iwork, int info );
        void cpbrfs( String uplo, int n, int kd, int nrhs,
                            IComplexNumber ab, int ldab,
                            IComplexNumber afb, int ldafb,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void zpbrfs( String uplo, int n, int kd, int nrhs,
                            IComplexNumber ab, int ldab,
                            IComplexNumber afb, int ldafb,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void sptrfs(int n, int nrhs,  long d,
                           long e,  long df,  long ef,
                           long b, int ldb, long x, int ldx,
                           long ferr, long berr, long work, int info );
        void dptrfs(int n, int nrhs,  long d,
                           long e,  long df,  long ef,
                           long b, int ldb, long x,int ldx, long ferr, long berr, long work,int info );
        void cptrfs( String uplo, int n, int nrhs,  long d,
                            IComplexNumber e,  long df,
                            IComplexNumber ef,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void zptrfs( String uplo, int n, int nrhs,
                            long d,  IComplexNumber e,
                            long df,  IComplexNumber ef,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void ssyrfs( String uplo, int n, int nrhs,  long a,int lda,  long af, int ldaf,int ipiv,  long b, int ldb,
                            long x, int ldx, long ferr, long berr,
                            long work, int iwork, int info );
        void dsyrfs( String uplo, int n, int nrhs,
                            long a, int lda,  long af,int ldaf,int ipiv,  long b,int ldb, long x, int ldx, long ferr,
                            long berr, long work, int iwork,int info );
        void csyrfs( String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void zsyrfs( String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void dsyrfsx( String uplo, String equed, int n, int nrhs,
                             long a, int lda,  long af,int ldaf,int ipiv,  long s,
                             long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams, long params,
                             long work, int iwork, int info );
        void ssyrfsx( String uplo, String equed, int n, int nrhs,
                             long a, int lda,  long af,int ldaf,int ipiv,  long s,
                             long b, int ldb, long x, int ldx,
                             long rcond, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params, long work,int iwork, int info );
        void zsyrfsx( String uplo, String equed, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv,  long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void csyrfsx( String uplo, String equed, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv,  long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long berr, int int_bnds, long erint_norm,
                             long erint_comp, int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void cherfs( String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void zherfs( String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void zherfsx( String uplo, String equed, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv,  long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void cherfsx( String uplo, String equed, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv,  long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long berr, int int_bnds, long erint_norm,
                             long erint_comp, int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void ssprfs( String uplo, int n, int nrhs,
                            long ap,  long afp,int ipiv,
                            long b, int ldb, long x, int ldx,
                            long ferr, long berr, long work, int iwork,int info );
        void dsprfs( String uplo, int n, int nrhs,
                            long ap,  long afp,int ipiv,
                            long b, int ldb, long x,int ldx, long ferr, long berr, long work,int iwork, int info );
        void csprfs( String uplo, int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber afp,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void zsprfs( String uplo, int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber afp,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void chprfs( String uplo, int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber afp,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void zhprfs( String uplo, int n, int nrhs,
                            IComplexNumber ap,
                            IComplexNumber afp,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void strrfs( String uplo, String trans, String diag, int n,int nrhs,  long a, int lda,
                            long b, int ldb,  long x,int ldx, long ferr, long berr, long work,int iwork, int info );
        void dtrrfs( String uplo, String trans, String diag, int n,int nrhs,  long a, int lda,
                            long b, int ldb,  long x,int ldx, long ferr, long berr, long work,int iwork, int info );
        void ctrrfs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber b,int ldb,  IComplexNumber x,int ldx, long ferr, long berr,
                            IComplexNumber work, long rwork,int info );
        void ztrrfs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber b,int ldb,  IComplexNumber x,int ldx, long ferr, long berr,
                            IComplexNumber work, long rwork,int info );
        void stprfs( String uplo, String trans, String diag, int n,int nrhs,  long ap,  long b,int ldb,  long x, int ldx,
                            long ferr, long berr, long work, int iwork,int info );
        void dtprfs( String uplo, String trans, String diag, int n,int nrhs,  long ap,  long b,int ldb,  long x, int ldx,
                            long ferr, long berr, long work, int iwork,int info );
        void ctprfs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber ap,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void ztprfs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber ap,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void stbrfs( String uplo, String trans, String diag, int n,int kd, int nrhs,  long ab,int ldab,  long b, int ldb,
                            long x, int ldx, long ferr, long berr,
                            long work, int iwork, int info );
        void dtbrfs( String uplo, String trans, String diag, int n,int kd, int nrhs,  long ab,int ldab,  long b, int ldb,
                            long x, int ldx, long ferr,
                            long berr, long work, int iwork,int info );
        void ctbrfs( String uplo, String trans, String diag, int n,int kd, int nrhs,
                            IComplexNumber ab, int ldab,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long ferr,
                            long berr, IComplexNumber work, long rwork,int info );
        void ztbrfs( String uplo, String trans, String diag, int n,int kd, int nrhs,
                            IComplexNumber ab, int ldab,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void sgetri(int n, long a, int lda,int ipiv, long work, int lwork,int info );
        void dgetri(int n, long a, int lda,int ipiv, long work, int lwork,int info );
        void cgetri(int n, IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork, int info );
        void zgetri(int n, IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork, int info );
        void spotri( String uplo, int n, long a, int lda,int info );
        void dpotri( String uplo, int n, long a, int lda,int info );
        void cpotri( String uplo, int n, IComplexNumber a,int lda, int info );
        void zpotri( String uplo, int n, IComplexNumber a,int lda, int info );
        void dpftri( String transr, String uplo, int n, long a,int info );
        void spftri( String transr, String uplo, int n, long a,int info );
        void zpftri( String transr, String uplo, int n,
                            IComplexNumber a, int info );
        void cpftri( String transr, String uplo, int n,
                            IComplexNumber a, int info );
        void spptri( String uplo, int n, long ap, int info );
        void dpptri( String uplo, int n, long ap, int info );
        void cpptri( String uplo, int n, IComplexNumber ap,int info );
        void zpptri( String uplo, int n, IComplexNumber ap,int info );
        void ssytri( String uplo, int n, long a, int lda,int ipiv, long work, int info );
        void dsytri( String uplo, int n, long a, int lda,int ipiv, long work, int info );
        void csytri( String uplo, int n, IComplexNumber a,int lda,int ipiv,
                            IComplexNumber work, int info );
        void zsytri( String uplo, int n, IComplexNumber a,int lda,int ipiv,
                            IComplexNumber work, int info );
        void chetri( String uplo, int n, IComplexNumber a,int lda,int ipiv,
                            IComplexNumber work, int info );
        void zhetri( String uplo, int n, IComplexNumber a,int lda,int ipiv,
                            IComplexNumber work, int info );
        void ssptri( String uplo, int n, long ap,int ipiv, long work, int info );
        void dsptri( String uplo, int n, long ap,int ipiv, long work, int info );
        void csptri( String uplo, int n, IComplexNumber ap,int ipiv, IComplexNumber work,int info );
        void zsptri( String uplo, int n, IComplexNumber ap,int ipiv, IComplexNumber work,int info );
        void chptri( String uplo, int n, IComplexNumber ap,int ipiv, IComplexNumber work,int info );
        void zhptri( String uplo, int n, IComplexNumber ap,int ipiv, IComplexNumber work,int info );
        void strtri( String uplo, String diag, int n, long a,int lda, int info );
        void dtrtri( String uplo, String diag, int n, long a,int lda, int info );
        void ctrtri( String uplo, String diag, int n,
                            IComplexNumber a, int lda,int info );
        void ztrtri( String uplo, String diag, int n,
                            IComplexNumber a, int lda,int info );
        void dtftri( String transr, String uplo, String diag, int n,
                            long a, int info );
        void stftri( String transr, String uplo, String diag, int n,
                            long a, int info );
        void ztftri( String transr, String uplo, String diag, int n,
                            IComplexNumber a, int info );
        void ctftri( String transr, String uplo, String diag, int n,
                            IComplexNumber a, int info );
        void stptri( String uplo, String diag, int n, long ap,int info );
        void dtptri( String uplo, String diag, int n, long ap,int info );
        void ctptri( String uplo, String diag, int n,
                            IComplexNumber ap, int info );
        void ztptri( String uplo, String diag, int n,
                            IComplexNumber ap, int info );
        void sgeequ(int m, int n,  long a,int lda, long r, long c, long rowcnd,
                           long colcnd, long amax, int info );
        void dgeequ(int m, int n,  long a,int lda, long r, long c, long rowcnd,
                           long colcnd, long amax, int info );
        void cgeequ(int m, int n,  IComplexNumber a,int lda, long r, long c, long rowcnd,
                           long colcnd, long amax, int info );
        void zgeequ(int m, int n,
                           IComplexNumber a, int lda, long r,
                           long c, long rowcnd, long colcnd, long amax,int info );
        void dgeequb(int m, int n,  long a,int lda, long r, long c, long rowcnd,
                            long colcnd, long amax, int info );
        void sgeequb(int m, int n,  long a,int lda, long r, long c, long rowcnd,
                            long colcnd, long amax, int info );
        void zgeequb(int m, int n,
                            IComplexNumber a, int lda, long r,
                            long c, long rowcnd, long colcnd, long amax,int info );
        void cgeequb(int m, int n,
                            IComplexNumber a, int lda, long r,
                            long c, long rowcnd, long colcnd, long amax,int info );
        void sgbequ(int m, int n, int kl,int ku,  long ab, int ldab, long r,
                           long c, long rowcnd, long colcnd, long amax,int info );
        void dgbequ(int m, int n, int kl,int ku,  long ab, int ldab,
                           long r, long c, long rowcnd, long colcnd,
                           long amax, int info );
        void cgbequ(int m, int n, int kl,int ku,  IComplexNumber ab,int ldab, long r, long c, long rowcnd,
                           long colcnd, long amax, int info );
        void zgbequ(int m, int n, int kl,int ku,  IComplexNumber ab,int ldab, long r, long c, long rowcnd,
                           long colcnd, long amax, int info );
        void dgbequb(int m, int n, int kl,int ku,  long ab, int ldab,
                            long r, long c, long rowcnd, long colcnd,
                            long amax, int info );
        void sgbequb(int m, int n, int kl,int ku,  long ab, int ldab,
                            long r, long c, long rowcnd, long colcnd,
                            long amax, int info );
        void zgbequb(int m, int n, int kl,int ku,  IComplexNumber ab,int ldab, long r, long c, long rowcnd,
                            long colcnd, long amax, int info );
        void cgbequb(int m, int n, int kl,int ku,  IComplexNumber ab,int ldab, long r, long c, long rowcnd,
                            long colcnd, long amax, int info );
        void spoequ(int n,  long a, int lda, long s,
                           long scond, long amax, int info );
        void dpoequ(int n,  long a, int lda, long s,
                           long scond, long amax, int info );
        void cpoequ(int n,  IComplexNumber a,int lda, long s, long scond, long amax,int info );
        void zpoequ(int n,  IComplexNumber a,int lda, long s, long scond, long amax,int info );
        void dpoequb(int n,  long a, int lda, long s,
                            long scond, long amax, int info );
        void spoequb(int n,  long a, int lda, long s,
                            long scond, long amax, int info );
        void zpoequb(int n,  IComplexNumber a,int lda, long s, long scond, long amax,int info );
        void cpoequb(int n,  IComplexNumber a,int lda, long s, long scond, long amax,int info );
        void sppequ( String uplo, int n,  long ap, long s,
                            long scond, long amax, int info );
        void dppequ( String uplo, int n,  long ap, long s,
                            long scond, long amax, int info );
        void cppequ( String uplo, int n,  IComplexNumber ap,
                            long s, long scond, long amax, int info );
        void zppequ( String uplo, int n,  IComplexNumber ap,
                            long s, long scond, long amax, int info );
        void spbequ( String uplo, int n, int kd,  long ab,int ldab, long s, long scond, long amax,int info );
        void dpbequ( String uplo, int n, int kd,  long ab,int ldab, long s, long scond, long amax,int info );
        void cpbequ( String uplo, int n, int kd,
                            IComplexNumber ab, int ldab, long s,
                            long scond, long amax, int info );
        void zpbequ( String uplo, int n, int kd,
                            IComplexNumber ab, int ldab,
                            long s, long scond, long amax, int info );
        void dsyequb( String uplo, int n,  long a,int lda, long s, long scond, long amax,
                             long work, int info );
        void ssyequb( String uplo, int n,  long a, int lda,
                             long s, long scond, long amax, long work,int info );
        void zsyequb( String uplo, int n,  IComplexNumber a,int lda, long s, long scond, long amax,
                             IComplexNumber work, int info );
        void csyequb( String uplo, int n,  IComplexNumber a,int lda, long s, long scond, long amax,
                             IComplexNumber work, int info );
        void zheequb( String uplo, int n,  IComplexNumber a,int lda, long s, long scond, long amax,
                             IComplexNumber work, int info );
        void cheequb( String uplo, int n,  IComplexNumber a,int lda, long s, long scond, long amax,
                             IComplexNumber work, int info );
        void sgesv(int n, int nrhs, long a, int lda,int ipiv, long b, int ldb,int info );
        void dgesv(int n, int nrhs, long a, int lda,int ipiv, long b, int ldb,int info );
        void cgesv(int n, int nrhs, IComplexNumber a,int lda, int ipiv, IComplexNumber b,int ldb, int info );
        void zgesv(int n, int nrhs, IComplexNumber a,int lda, int ipiv, IComplexNumber b,int ldb, int info );
        void dsgesv(int n, int nrhs, long a, int lda,int ipiv, long b, int ldb, long x,int ldx, long work, long swork,int iter, int info );
        void zcgesv(int n, int nrhs, IComplexNumber a,int lda, int ipiv, IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                           IComplexNumber work, IComplexNumber swork,
                           long rwork, int iter, int info );
        void sgesvx( String fact, String trans, int n, int nrhs,
                            long a, int lda, long af, int ldaf,int ipiv, String equed, long r, long c, long b,int ldb, long x, int ldx, long rcond,
                            long ferr, long berr, long work, int iwork,int info );
        void dgesvx( String fact, String trans, int n, int nrhs,
                            long a, int lda, long af, int ldaf,int ipiv, String equed, long r, long c,
                            long b, int ldb, long x, int ldx,
                            long rcond, long ferr, long berr, long work,int iwork, int info );
        void cgesvx( String fact, String trans, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long rcond,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void zgesvx( String fact, String trans, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long rcond,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void dgesvxx( String fact, String trans, int n, int nrhs,
                             long a, int lda, long af, int ldaf,int ipiv, String equed, long r, long c,
                             long b, int ldb, long x, int ldx,
                             long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams, long params,
                             long work, int iwork, int info );
        void sgesvxx( String fact, String trans, int n, int nrhs,
                             long a, int lda, long af, int ldaf,int ipiv, String equed, long r, long c,
                             long b, int ldb, long x, int ldx,
                             long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams, long params,
                             long work, int iwork, int info );
        void zgesvxx( String fact, String trans, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long rpvgrw, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void cgesvxx( String fact, String trans, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long rpvgrw, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void sgbsv(int n, int kl, int ku,int nrhs, long ab, int ldab,int ipiv, long b, int ldb,int info );
        void dgbsv(int n, int kl, int ku,int nrhs, long ab, int ldab,int ipiv, long b, int ldb,int info );
        void cgbsv(int n, int kl, int ku,int nrhs, IComplexNumber ab, int ldab,int ipiv, IComplexNumber b, int ldb,int info );
        void zgbsv(int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, int ipiv, IComplexNumber b,int ldb, int info );
        void sgbsvx( String fact, String trans, int n, int kl,int ku, int nrhs, long ab,int ldab, long afb, int ldafb,int ipiv, String equed, long r, long c, long b,int ldb, long x, int ldx, long rcond,
                            long ferr, long berr, long work, int iwork,int info );
        void dgbsvx( String fact, String trans, int n, int kl,int ku, int nrhs, long ab,int ldab, long afb, int ldafb,int ipiv, String equed, long r, long c,
                            long b, int ldb, long x, int ldx,
                            long rcond, long ferr, long berr, long work,int iwork, int info );
        void cgbsvx( String fact, String trans, int n, int kl,int ku, int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed, long r,
                            long c, IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long rcond,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void zgbsvx( String fact, String trans, int n, int kl,int ku, int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed, long r,
                            long c, IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long rcond,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void dgbsvxx( String fact, String trans, int n, int kl,int ku, int nrhs, long ab,int ldab, long afb, int ldafb,int ipiv, String equed, long r, long c,
                             long b, int ldb, long x, int ldx,
                             long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams, long params,
                             long work, int iwork, int info );
        void sgbsvxx( String fact, String trans, int n, int kl,int ku, int nrhs, long ab,int ldab, long afb, int ldafb,int ipiv, String equed, long r, long c,
                             long b, int ldb, long x, int ldx,
                             long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams, long params,
                             long work, int iwork, int info );
        void zgbsvxx( String fact, String trans, int n, int kl,int ku, int nrhs,
                             IComplexNumber ab, int ldab,
                             IComplexNumber afb, int ldafb,int ipiv, String equed, long r, long c,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long rpvgrw, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void cgbsvxx( String fact, String trans, int n, int kl,int ku, int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed, long r,
                             long c, IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long rpvgrw, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void sgtsv(int n, int nrhs, long dl, long d,
                          long du, long b, int ldb, int info );
        void dgtsv(int n, int nrhs, long dl, long d,
                          long du, long b, int ldb, int info );
        void cgtsv(int n, int nrhs, IComplexNumber dl,
                          IComplexNumber d, IComplexNumber du,
                          IComplexNumber b, int ldb, int info );
        void zgtsv(int n, int nrhs, IComplexNumber dl,
                          IComplexNumber d, IComplexNumber du,
                          IComplexNumber b, int ldb,int info );
        void sgtsvx( String fact, String trans, int n, int nrhs,
                            long dl,  long d,  long du,
                            long dlf, long df, long duf, long du2,int ipiv,  long b, int ldb, long x,int ldx, long rcond, long ferr, long berr,
                            long work, int iwork, int info );
        void dgtsvx( String fact, String trans, int n, int nrhs,
                            long dl,  long d,  long du,
                            long dlf, long df, long duf, long du2,int ipiv,  long b, int ldb,
                            long x, int ldx, long rcond, long ferr,
                            long berr, long work, int iwork,int info );
        void cgtsvx( String fact, String trans, int n, int nrhs,
                            IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du, IComplexNumber dlf,
                            IComplexNumber df, IComplexNumber duf,
                            IComplexNumber du2, int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long rcond,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void zgtsvx( String fact, String trans, int n, int nrhs,
                            IComplexNumber dl,
                            IComplexNumber d,
                            IComplexNumber du, IComplexNumber dlf,
                            IComplexNumber df, IComplexNumber duf,
                            IComplexNumber du2, int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long rcond,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void sposv( String uplo, int n, int nrhs, long a,int lda, long b, int ldb,int info );
        void dposv( String uplo, int n, int nrhs, long a,int lda, long b, int ldb,int info );
        void cposv( String uplo, int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb, int info );
        void zposv( String uplo, int n, int nrhs,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,int info );
        void dsposv( String uplo, int n, int nrhs, long a,int lda, long b, int ldb, long x,int ldx, long work, long swork,int iter, int info );
        void zcposv( String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx,
                            IComplexNumber work, IComplexNumber swork,
                            long rwork, int iter, int info );
        void sposvx( String fact, String uplo, int n, int nrhs,
                            long a, int lda, long af, int ldaf,
                            String equed, long s, long b, int ldb, long x,int ldx, long rcond, long ferr, long berr,
                            long work, int iwork, int info );
        void dposvx( String fact, String uplo, int n, int nrhs,
                            long a, int lda, long af, int ldaf,
                            String equed, long s, long b, int ldb,
                            long x, int ldx, long rcond, long ferr,
                            long berr, long work, int iwork,int info );
        void cposvx( String fact, String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf, String equed,
                            long s, IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long rcond,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void zposvx( String fact, String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf, String equed,
                            long s, IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long rcond,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void dposvxx( String fact, String uplo, int n, int nrhs,
                             long a, int lda, long af, int ldaf,
                             String equed, long s, long b, int ldb,
                             long x, int ldx, long rcond, long rpvgrw,
                             long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params, long work,int iwork, int info );
        void sposvxx( String fact, String uplo, int n, int nrhs,
                             long a, int lda, long af, int ldaf,
                             String equed, long s, long b, int ldb, long x,int ldx, long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                             long erint_comp, int nparams, long params,
                             long work, int iwork, int info );
        void zposvxx( String fact, String uplo, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf, String equed,
                             long s, IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long rpvgrw, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void cposvxx( String fact, String uplo, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf, String equed,
                             long s, IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long rpvgrw, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void sppsv( String uplo, int n, int nrhs, long ap,
                           long b, int ldb, int info );
        void dppsv( String uplo, int n, int nrhs, long ap,
                           long b, int ldb, int info );
        void cppsv( String uplo, int n, int nrhs,
                           IComplexNumber ap, IComplexNumber b,int ldb, int info );
        void zppsv( String uplo, int n, int nrhs,
                           IComplexNumber ap, IComplexNumber b,int ldb, int info );
        void sppsvx( String fact, String uplo, int n, int nrhs,
                            long ap, long afp, String equed, long s, long b,int ldb, long x, int ldx, long rcond,
                            long ferr, long berr, long work, int iwork,int info );
        void dppsvx( String fact, String uplo, int n, int nrhs,
                            long ap, long afp, String equed, long s, long b,int ldb, long x, int ldx, long rcond,
                            long ferr, long berr, long work, int iwork,int info );
        void cppsvx( String fact, String uplo, int n, int nrhs,
                            IComplexNumber ap, IComplexNumber afp,
                            String equed, long s, IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork,int info );
        void zppsvx( String fact, String uplo, int n, int nrhs,
                            IComplexNumber ap, IComplexNumber afp,
                            String equed, long s, IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork,int info );
        void spbsv( String uplo, int n, int kd, int nrhs,
                           long ab, int ldab, long b, int ldb,int info );
        void dpbsv( String uplo, int n, int kd, int nrhs,
                           long ab, int ldab, long b, int ldb,int info );
        void cpbsv( String uplo, int n, int kd, int nrhs,
                           IComplexNumber ab, int ldab,
                           IComplexNumber b, int ldb, int info );
        void zpbsv( String uplo, int n, int kd, int nrhs,
                           IComplexNumber ab, int ldab,
                           IComplexNumber b, int ldb,int info );
        void spbsvx( String fact, String uplo, int n, int kd,int nrhs, long ab, int ldab, long afb,int ldafb, String equed, long s, long b,int ldb, long x, int ldx, long rcond,
                            long ferr, long berr, long work, int iwork,int info );
        void dpbsvx( String fact, String uplo, int n, int kd,int nrhs, long ab, int ldab, long afb,int ldafb, String equed, long s, long b,int ldb, long x, int ldx, long rcond,
                            long ferr, long berr, long work, int iwork,int info );
        void cpbsvx( String fact, String uplo, int n, int kd,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, String equed, long s,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long rcond,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void zpbsvx( String fact, String uplo, int n, int kd,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, String equed, long s,
                            IComplexNumber b, int ldb,
                            IComplexNumber x, int ldx, long rcond,
                            long ferr, long berr, IComplexNumber work,
                            long rwork, int info );
        void sptsv(int n, int nrhs, long d, long e,
                          long b, int ldb, int info );
        void dptsv(int n, int nrhs, long d, long e,
                          long b, int ldb, int info );
        void cptsv(int n, int nrhs, long d,
                          IComplexNumber e, IComplexNumber b,int ldb, int info );
        void zptsv(int n, int nrhs, long d,
                          IComplexNumber e, IComplexNumber b,int ldb, int info );
        void sptsvx( String fact, int n, int nrhs,  long d,
                            long e, long df, long ef,  long b,int ldb, long x, int ldx, long rcond,
                            long ferr, long berr, long work, int info );
        void dptsvx( String fact, int n, int nrhs,
                            long d,  long e, long df, long ef,
                            long b, int ldb, long x,int ldx, long rcond, long ferr, long berr,
                            long work, int info );
        void cptsvx( String fact, int n, int nrhs,  long d,
                            IComplexNumber e, long df,
                            IComplexNumber ef,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork,int info );
        void zptsvx( String fact, int n, int nrhs,
                            long d,  IComplexNumber e, long df,
                            IComplexNumber ef,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork,int info );
        void ssysv( String uplo, int n, int nrhs, long a,int lda, int ipiv, long b, int ldb,
                           long work, int lwork, int info );
        void dsysv( String uplo, int n, int nrhs, long a,int lda, int ipiv, long b,int ldb, long work, int lwork,int info );
        void csysv( String uplo, int n, int nrhs,
                           IComplexNumber a, int lda, int ipiv,
                           IComplexNumber b, int ldb,
                           IComplexNumber work, int lwork,int info );
        void zsysv( String uplo, int n, int nrhs,
                           IComplexNumber a, int lda, int ipiv,
                           IComplexNumber b, int ldb,
                           IComplexNumber work, int lwork,int info );
        void ssysvx( String fact, String uplo, int n, int nrhs,
                            long a, int lda, long af,int ldaf, int ipiv,  long b,int ldb, long x, int ldx, long rcond,
                            long ferr, long berr, long work, int lwork,int iwork, int info );
        void dsysvx( String fact, String uplo, int n, int nrhs,
                            long a, int lda, long af,int ldaf, int ipiv,  long b,int ldb, long x, int ldx, long rcond,
                            long ferr, long berr, long work, int lwork,int iwork, int info );
        void csysvx( String fact, String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, int lwork, long rwork,int info );
        void zsysvx( String fact, String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, int lwork,
                            long rwork, int info );
        void dsysvxx( String fact, String uplo, int n, int nrhs,
                             long a, int lda, long af, int ldaf,int ipiv, String equed, long s, long b,int ldb, long x, int ldx, long rcond,
                             long rpvgrw, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params, long work,int iwork, int info );
        void ssysvxx( String fact, String uplo, int n, int nrhs,
                             long a, int lda, long af, int ldaf,int ipiv, String equed, long s, long b,int ldb, long x, int ldx, long rcond,
                             long rpvgrw, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params, long work,int iwork, int info );
        void zsysvxx( String fact, String uplo, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long rpvgrw, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void csysvxx( String fact, String uplo, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long rpvgrw, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void chesv( String uplo, int n, int nrhs,
                           IComplexNumber a, int lda, int ipiv,
                           IComplexNumber b, int ldb,
                           IComplexNumber work, int lwork,int info );
        void zhesv( String uplo, int n, int nrhs,
                           IComplexNumber a, int lda, int ipiv,
                           IComplexNumber b, int ldb,
                           IComplexNumber work, int lwork,int info );
        void chesvx( String fact, String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, int lwork, long rwork,int info );
        void zhesvx( String fact, String uplo, int n, int nrhs,
                            IComplexNumber a, int lda,
                            IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, int lwork,
                            long rwork, int info );
        void zhesvxx( String fact, String uplo, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long rpvgrw, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void chesvxx( String fact, String uplo, int n, int nrhs,
                             IComplexNumber a, int lda,
                             IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                             IComplexNumber b, int ldb,
                             IComplexNumber x, int ldx, long rcond,
                             long rpvgrw, long berr, int int_bnds,
                             long erint_norm, long erint_comp,int nparams, long params,
                             IComplexNumber work, long rwork,int info );
        void sspsv( String uplo, int n, int nrhs, long ap,int ipiv, long b, int ldb,int info );
        void dspsv( String uplo, int n, int nrhs, long ap,int ipiv, long b, int ldb,int info );
        void cspsv( String uplo, int n, int nrhs,
                           IComplexNumber ap, int ipiv,
                           IComplexNumber b, int ldb, int info );
        void zspsv( String uplo, int n, int nrhs,
                           IComplexNumber ap, int ipiv,
                           IComplexNumber b, int ldb,int info );
        void sspsvx( String fact, String uplo, int n, int nrhs,
                            long ap, long afp, int ipiv,
                            long b, int ldb, long x, int ldx,
                            long rcond, long ferr, long berr, long work,int iwork, int info );
        void dspsvx( String fact, String uplo, int n, int nrhs,
                            long ap, long afp, int ipiv,
                            long b, int ldb, long x,int ldx, long rcond, long ferr, long berr,
                            long work, int iwork, int info );
        void cspsvx( String fact, String uplo, int n, int nrhs,
                            IComplexNumber ap, IComplexNumber afp,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork,int info );
        void zspsvx( String fact, String uplo, int n, int nrhs,
                            IComplexNumber ap, IComplexNumber afp,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork,int info );
        void chpsv( String uplo, int n, int nrhs,
                           IComplexNumber ap, int ipiv,
                           IComplexNumber b, int ldb, int info );
        void zhpsv( String uplo, int n, int nrhs,
                           IComplexNumber ap, int ipiv,
                           IComplexNumber b, int ldb,int info );
        void chpsvx( String fact, String uplo, int n, int nrhs,
                            IComplexNumber ap, IComplexNumber afp,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork,int info );
        void zhpsvx( String fact, String uplo, int n, int nrhs,
                            IComplexNumber ap, IComplexNumber afp,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                            long rcond, long ferr, long berr,
                            IComplexNumber work, long rwork,int info );
        void sgeqrf(int m, int n, long a, int lda,
                           long tau, long work, int lwork,int info );
        void dgeqrf(int m, int n, long a, int lda,
                           long tau, long work, int lwork,int info );
        void cgeqrf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                           IComplexNumber work, int lwork,int info );
        void zgeqrf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                           IComplexNumber work, int lwork,int info );
        void sgeqpf(int m, int n, long a, int lda,int jpvt, long tau, long work,int info );
        void dgeqpf(int m, int n, long a, int lda,int jpvt, long tau, long work,int info );
        void cgeqpf(int m, int n, IComplexNumber a,int lda, int jpvt,
                           IComplexNumber tau, IComplexNumber work,
                           long rwork, int info );
        void zgeqpf(int m, int n, IComplexNumber a,int lda, int jpvt,
                           IComplexNumber tau, IComplexNumber work,
                           long rwork, int info );
        void sgeqp3(int m, int n, long a, int lda,int jpvt, long tau, long work,int lwork, int info );
        void dgeqp3(int m, int n, long a, int lda,int jpvt, long tau, long work,int lwork, int info );
        void cgeqp3(int m, int n, IComplexNumber a,int lda, int jpvt,
                           IComplexNumber tau, IComplexNumber work,int lwork, long rwork, int info );
        void zgeqp3(int m, int n, IComplexNumber a,int lda, int jpvt,
                           IComplexNumber tau, IComplexNumber work,int lwork, long rwork, int info );
        void sorgqr(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
        void dorgqr(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
        void sormqr( String side, String trans, int m, int n,int k,  long a, int lda,
                            long tau, long c, int ldc, long work,int lwork, int info );
        void dormqr( String side, String trans, int m, int n,int k,  long a, int lda,
                            long tau, long c, int ldc, long work,int lwork, int info );
        void cungqr(int m, int n, int k,
                           IComplexNumber a, int lda,
                           IComplexNumber tau, IComplexNumber work,int lwork, int info );
        void zungqr(int m, int n, int k,
                           IComplexNumber a, int lda,
                           IComplexNumber tau,
                           IComplexNumber work, int lwork,int info );
        void cunmqr( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork,int info );
        void zunmqr( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork,int info );
        void sgelqf(int m, int n, long a, int lda,
                           long tau, long work, int lwork,int info );
        void dgelqf(int m, int n, long a, int lda,
                           long tau, long work, int lwork,int info );
        void cgelqf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                           IComplexNumber work, int lwork,int info );
        void zgelqf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                           IComplexNumber work, int lwork,int info );
        void sorglq(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
        void dorglq(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
        void sormlq( String side, String trans, int m, int n,int k,  long a, int lda,
                            long tau, long c, int ldc, long work,int lwork, int info );
        void dormlq( String side, String trans, int m, int n,int k,  long a, int lda,
                            long tau, long c, int ldc, long work,int lwork, int info );
        void cunglq(int m, int n, int k,
                           IComplexNumber a, int lda,
                           IComplexNumber tau, IComplexNumber work,int lwork, int info );
        void zunglq(int m, int n, int k,
                           IComplexNumber a, int lda,
                           IComplexNumber tau,
                           IComplexNumber work, int lwork,int info );
        void cunmlq( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork,int info );
        void zunmlq( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork,int info );
        void sgeqlf(int m, int n, long a, int lda,
                           long tau, long work, int lwork,int info );
        void dgeqlf(int m, int n, long a, int lda,
                           long tau, long work, int lwork,int info );
        void cgeqlf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                           IComplexNumber work, int lwork,int info );
        void zgeqlf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                           IComplexNumber work, int lwork,int info );
        void sorgql(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
        void dorgql(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
        void cungql(int m, int n, int k,
                           IComplexNumber a, int lda,
                           IComplexNumber tau, IComplexNumber work,int lwork, int info );
        void zungql(int m, int n, int k,
                           IComplexNumber a, int lda,
                           IComplexNumber tau,
                           IComplexNumber work, int lwork,int info );
        void sormql( String side, String trans, int m, int n,int k,  long a, int lda,
                            long tau, long c, int ldc, long work,int lwork, int info );
        void dormql( String side, String trans, int m, int n,int k,  long a, int lda,
                            long tau, long c, int ldc, long work,int lwork, int info );
        void cunmql( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork,int info );
        void zunmql( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork,int info );
        void sgerqf(int m, int n, long a, int lda,
                           long tau, long work, int lwork,int info );
        void dgerqf(int m, int n, long a, int lda,
                           long tau, long work, int lwork,int info );
        void cgerqf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                           IComplexNumber work, int lwork,int info );
        void zgerqf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                           IComplexNumber work, int lwork,int info );
        void sorgrq(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
        void dorgrq(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
        void cungrq(int m, int n, int k,
                           IComplexNumber a, int lda,
                           IComplexNumber tau, IComplexNumber work,int lwork, int info );
        void zungrq(int m, int n, int k,
                           IComplexNumber a, int lda,
                           IComplexNumber tau,
                           IComplexNumber work, int lwork,int info );
        void sormrq( String side, String trans, int m, int n,int k,  long a, int lda,
                            long tau, long c, int ldc, long work,int lwork, int info );
        void dormrq( String side, String trans, int m, int n,int k,  long a, int lda,
                            long tau, long c, int ldc, long work,int lwork, int info );
        void cunmrq( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork,int info );
        void zunmrq( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork,int info );
        void stzrzf(int m, int n, long a, int lda,
                           long tau, long work, int lwork,int info );
        void dtzrzf(int m, int n, long a, int lda,
                           long tau, long work, int lwork,int info );
        void ctzrzf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                           IComplexNumber work, int lwork,int info );
        void ztzrzf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                           IComplexNumber work, int lwork,int info );
        void sormrz( String side, String trans, int m, int n,int k, int l,  long a,int lda,  long tau, long c,int ldc, long work, int lwork,int info );
        void dormrz( String side, String trans, int m, int n,int k, int l,  long a,int lda,  long tau, long c,int ldc, long work, int lwork,int info );
        void cunmrz( String side, String trans, int m, int n,int k, int l,  IComplexNumber a,int lda,  IComplexNumber tau,
                            IComplexNumber c, int ldc,
                            IComplexNumber work, int lwork,int info );
        void zunmrz( String side, String trans, int m, int n,int k, int l,
                            IComplexNumber a, int lda,
                            IComplexNumber tau, IComplexNumber c,int ldc, IComplexNumber work,int lwork, int info );
        void sggqrf(int n, int m, int p, long a,int lda, long taua, long b, int ldb,
                           long taub, long work, int lwork,int info );
        void dggqrf(int n, int m, int p, long a,int lda, long taua, long b, int ldb,
                           long taub, long work, int lwork,int info );
        void cggqrf(int n, int m, int p,
                           IComplexNumber a, int lda,
                           IComplexNumber taua, IComplexNumber b,int ldb, IComplexNumber taub,
                           IComplexNumber work, int lwork,int info );
        void zggqrf(int n, int m, int p,
                           IComplexNumber a, int lda,
                           IComplexNumber taua, IComplexNumber b,int ldb, IComplexNumber taub,
                           IComplexNumber work, int lwork,int info );
        void sggrqf(int m, int p, int n, long a,int lda, long taua, long b, int ldb,
                           long taub, long work, int lwork,int info );
        void dggrqf(int m, int p, int n, long a,int lda, long taua, long b, int ldb,
                           long taub, long work, int lwork,int info );
        void cggrqf(int m, int p, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber taua, IComplexNumber b,int ldb, IComplexNumber taub,
                           IComplexNumber work, int lwork,int info );
        void zggrqf(int m, int p, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber taua, IComplexNumber b,int ldb, IComplexNumber taub,
                           IComplexNumber work, int lwork,int info );
        void sgebrd(int m, int n, long a, int lda,
                           long d, long e, long tauq, long taup, long work,int lwork, int info );
        void dgebrd(int m, int n, long a, int lda,
                           long d, long e, long tauq, long taup,
                           long work, int lwork, int info );
        void cgebrd(int m, int n, IComplexNumber a,int lda, long d, long e,
                           IComplexNumber tauq, IComplexNumber taup,
                           IComplexNumber work, int lwork,int info );
        void zgebrd(int m, int n, IComplexNumber a,int lda, long d, long e,
                           IComplexNumber tauq, IComplexNumber taup,
                           IComplexNumber work, int lwork,int info );
        void sgbbrd( String vect, int m, int n, int ncc,int kl, int ku, long ab, int ldab,
                            long d, long e, long q, int ldq, long pt,int ldpt, long c, int ldc, long work,int info );
    }
};
#endif //LIBND4J_NATIVELAPACK_H_H
