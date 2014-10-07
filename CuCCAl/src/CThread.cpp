#include "CThread.h"


CThreadException::CThreadException( const string &sMessage, bool blSysMsg /*= false*/ ) throw() :m_sMsg(sMessage)
{
    if (blSysMsg) {
        m_sMsg.append(": ");
        m_sMsg.append(strerror(errno));
    }
}

CThreadException::~CThreadException() throw ()
{

}

CThread::CThread()
{

}

CThread::~CThread()
{

}

void CThread::Start() throw(CThreadException)
{
   CreateThread();
}

void CThread::Join() throw(CThreadException)
{
    int rc = pthread_join(m_Tid, NULL);
    if ( rc != 0 )
    {
        throw CThreadException("Error in thread join.... (pthread_join())", true);
    }
}

void* CThread::ThreadFunc( void* pTr )
{

    CThread* pThis = static_cast<CThread*>(pTr);
    pThis->Run();
    pthread_exit(0);
}

void CThread::CreateThread() throw(CThreadException)
{
    int rc = pthread_create (&m_Tid, NULL, ThreadFunc,this);
    if ( rc != 0 )
    {
        throw CThreadException("Error in thread creation... (pthread_create())", true);
    }
}
